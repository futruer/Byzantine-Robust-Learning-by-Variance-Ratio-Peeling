from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import copy
import config

class Client:

    def __init__(self, model:nn.Module = None,
                       data:torch.utils.data.DataLoader = None,
                       reform_func:callable = None,
                       id:int = None,
                       is_attacker:bool = False):
        self.model = model
        self.data = data
        self.reform_func = reform_func
        self.id = id
        self.is_normal = True
        self.is_attacker = is_attacker

    def get_XY_tlide(self, init_theta:torch.Tensor) -> None:
        
        # 期望labels_features为[label:torch.Tensor, features:torch.Tensor]其中前者一维tensor，后者二维tensor
        labels_features = self.model.extract_features(self.data)
        
        # 期望df的每一行代表一个样本，第一列代表Y_tlide，其余列代表X_tlide的各个特征
        # XY_tlide为pd.DataFrame类型
        XY_tlide = self.reform_func(labels_features, init_theta)
        self.XY_tlide = XY_tlide

    def XY_tlide_least_square(self, init_theta:torch.Tensor) -> torch.Tensor:
        self.get_XY_tlide(init_theta)
        
        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)
        
        # 添加截距项 (全1列)
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)
        
        # 计算最小二乘系数: β = (X^T * X)^(-1) * X^T * Y
        XTX = torch.matmul(X_tlide_with_intercept.T, X_tlide_with_intercept)
        XTY = torch.matmul(X_tlide_with_intercept.T, Y_tlide.unsqueeze(1))
        
        # 使用伪逆来避免奇异矩阵问题
        try:
            coefficients = torch.matmul(torch.linalg.pinv(XTX), XTY)
        except:
            # 如果 pinv 不可用，使用 solve
            coefficients = torch.linalg.solve(XTX, XTY)
        
        return coefficients.squeeze()
    
    def compute_mse_list(self, LS_coefficients: List[dict]) -> None:
        """
        对于传入的每一个最小二乘系数 beta_k，计算当前客户端数据下
        残差向量 Y_tlide - X_tlide^T beta_k 的 L2 范数，并绑定对应的 client id。
        LS_coefficients 是一个字典列表，每个字典包含 {"id": client_id, "beta_k": torch.Tensor}
        """
        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)  # (n,)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)  # (n, d)

        # 添加截距项 (全1列) -> X_with_intercept: (n, d+1)
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)

        # 将所有 beta_k 堆叠成矩阵 B: (d+1, K)，同时记录对应的 client id
        betas = []
        client_ids = []
        for coeff_dict in LS_coefficients:
            beta = coeff_dict["beta_k"]
            client_id = coeff_dict["id"]
            # 确保是一维向量形状 (d+1,)
            if beta.dim() > 1:
                beta = beta.view(-1)
            betas.append(beta)
            client_ids.append(client_id)
        B = torch.stack(betas, dim=1)  # (d+1, K)

        # 预测值矩阵: (n, K)
        preds = X_tlide_with_intercept @ B  # 矩阵乘法

        # 残差矩阵: (n, K)
        residuals = Y_tlide.unsqueeze(1) - preds

        # 对每个 beta_k 计算 L2 范数: 先按样本维度求平方和，再开方 -> (K,)
        l2_norms = torch.sqrt(torch.sum(residuals ** 2, dim=0))

        # 将 L2 范数与对应的 client id 绑定
        self.mse_list = []
        for i, client_id in enumerate(client_ids):
            self.mse_list.append({
                "id": client_id,
                "mse": l2_norms[i]
            })

    def compute_rest_mse_med(self, normal_client_ids:list[int]) -> torch.Tensor:
        """
        从 mse_list 中挑选出 normal_client_ids 对应的元素，
        提取这些元素的 mse 值，计算中位数并返回。
        """
        # 从 mse_list 中筛选出 id 在 normal_client_ids 中的元素
        filtered_mse_values = []
        for mse_item in self.mse_list:
            if mse_item["id"] in normal_client_ids:
                filtered_mse_values.append(mse_item["mse"])
        
        # 如果没有找到任何匹配的元素，返回 None 或抛出异常
        if len(filtered_mse_values) == 0:
            raise ValueError(f"No matching mse values found for normal_client_ids: {normal_client_ids}")
        
        # 将所有 mse 值堆叠成 tensor
        mse_tensor = torch.stack(filtered_mse_values)
        
        # 计算中位数
        mse_med = torch.median(mse_tensor)
        
        return mse_med

    def local_train(self, base_model: nn.Module, epochs: int, learning_rate: float,
                    gradient_attack_func: callable = None) -> nn.Module:
        """
        本地训练方法。

        参数:
            base_model: 基础模型（全局模型）
            epochs: 本地训练轮数
            learning_rate: 学习率
            gradient_attack_func: 梯度攻击函数（可选）

        返回:
            训练后的模型副本
        """
        import torch.nn.functional as F

        # 为当前客户端复制模型
        client_model = copy.deepcopy(base_model)
        device = next(base_model.parameters()).device
        client_model = client_model.to(device)
        client_model.train()

        optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for batch_data in self.data:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, labels = batch_data
                else:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = client_model(inputs)
                loss = F.cross_entropy(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 如果是攻击者且有梯度攻击函数，在计算梯度后应用攻击
                if gradient_attack_func is not None and self.is_attacker:
                    for param in client_model.parameters():
                        if param.grad is not None:
                            if gradient_attack_func.__name__ == 'gaussian_attack':
                                noise = torch.randn_like(param.grad) * 1.0
                                param.grad = param.grad + noise
                            elif gradient_attack_func.__name__ == 'sign_flipping_attack':
                                param.grad = param.grad * (-1.0)
                            elif gradient_attack_func.__name__ == 'scaling_attack':
                                param.grad = param.grad * 10.0

                optimizer.step()

        return client_model

    def train(self, local_epochs: int = 1, learning_rate: float = 0.1) -> dict:
        """
        使用当前 client 的数据进行本地训练 (FedAvg 模式)。
        返回包含 client id 和伪梯度 (初始权重 - 更新后权重) 的字典。
        """
        if self.model is None or self.data is None:
            raise ValueError("Model or Data is not set.")

        self.model.train()
        device = next(self.model.parameters()).device

        # 保存本轮初始权重
        original_weights = {name: param.data.clone() for name, param in self.model.named_parameters()}

        # 初始化优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        total_loss = 0.0
        num_batches = 0

        # 本地多 Epoch 迭代
        for epoch in range(local_epochs):
            for batch_data in self.data:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, labels = batch_data
                elif isinstance(batch_data, dict):
                    inputs = batch_data.get('input', batch_data.get('x', None))
                    labels = batch_data.get('label', batch_data.get('y', None))
                else:
                    inputs = batch_data
                    labels = None

                if inputs is not None:
                    inputs = inputs.to(device)
                if labels is not None:
                    labels = labels.to(device)

                optimizer.zero_grad()
                
                if labels is not None:
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels.long())
                else:
                    outputs = self.model(inputs)
                    loss = outputs.mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # 计算伪梯度 (Pseudo-gradients): 初始权重 - 更新后的权重
        pseudo_gradients = {}
        for name, param in self.model.named_parameters():
            pseudo_gradients[name] = original_weights[name] - param.data.clone()

        # 将模型恢复到初始状态，避免影响 Orchestrator 端的模型
        self.model.load_state_dict(original_weights)

        return {
            "id": self.id,
            "gradients": pseudo_gradients,
            "loss": total_loss / num_batches if num_batches > 0 else 0.0
        }

class Orchestrator:

    def __init__(self, model:nn.Module, clients:list[Client], gradient_attack_func: callable = None):
        self.model = model
        self.clients = clients

        self.init_normal_client_ids()

        # 筛选阶段的本地训练（应用梯度攻击）
        self.compute_init_theta(gradient_attack_func)
        self.compute_init_client_mse_list()

    def init_normal_client_ids(self) -> None:
        self.normal_client_dict_list: list[dict] = []
        for client in self.clients:
            self.normal_client_dict_list.append({
                "id": client.id,
                "is_normal": client.is_normal
            })

    def compute_init_theta(self, gradient_attack_func: callable = None) -> None:
        """
        计算初始估计 init_theta：
        1. 为每个客户端复制模型
        2. 每个客户端本地训练 INIT_EPOCHS 轮
        3. (可选) 对攻击者的梯度/参数应用攻击
        4. 收集 fc2 的权重参数，取坐标级中位数得到 init_theta
        5. 使用坐标级中位数聚合所有层的参数，用于筛选后的重新训练

        参数:
            gradient_attack_func: 梯度层面的攻击函数
        """
        print("\n[Init Theta] Starting local training on all clients...")

        # 收集每个客户端训练后的 fc2 权重
        fc2_weights = []

        for client in self.clients:
            # 调用 Client 类的本地训练方法
            # 梯度攻击在本地训练的每一轮中自动应用
            client_model = client.local_train(
                base_model=self.model,
                epochs=config.INIT_EPOCHS,
                learning_rate=config.INIT_LR,
                gradient_attack_func=gradient_attack_func
            )

            # 取出 fc2 的权重
            fc2_weight = client_model.fc2.weight.data.clone()
            fc2_weights.append(fc2_weight)

        print(f"[Init Theta] Local training completed for {len(self.clients)} clients")

        # 堆叠所有 fc2 权重: (num_clients, 10, 50)
        stacked_weights = torch.stack(fc2_weights, dim=0)

        # 计算坐标级中位数得到 init_theta
        init_theta = torch.median(stacked_weights, dim=0).values  # (10, 50)

        print(f"[Init Theta] init_theta shape: {init_theta.shape}")

        self.init_theta = init_theta

    def collect_client_betas(self) -> None:
        self.LS_coefficients: List[torch.Tensor] = []
        for client in self.clients:
            self.LS_coefficients.append({
                "id": client.id,
                "beta_k": client.XY_tlide_least_square(self.init_theta)
            })
        
    def compute_init_client_mse_list(self) -> None:
        self.collect_client_betas()
        for client in self.clients:
            client.compute_mse_list(self.LS_coefficients)

    def collect_client_mse_med(self) -> list[dict]:
        """
        将 normal_client_ids 发送到每个 Client，
        各个 Client 挑选出正常 client 对应的 mse_list 中的元素并取中位数，
        收集所有 Client 返回的中位数，形成字典列表。
        """
        # 从 normal_client_dict_list 中提取所有 is_normal=True 的 client id
        normal_client_ids = []
        for client_dict in self.normal_client_dict_list:
            if client_dict["is_normal"]:
                normal_client_ids.append(client_dict["id"])
        
        # 将 normal_client_ids 发送到每个 Client，收集返回的中位数
        result_list = []
        for client in self.clients:
            mse_med = client.compute_rest_mse_med(normal_client_ids)
            result_list.append({
                "id": client.id,
                "mse_med": mse_med
            })
        
        return result_list

    def update_normal_clients_by_gap(self, mse_med_list: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据收集到的所有 Client 返回的中位数，对这些中位数进行排序，
        排序后进行一阶差分，找到最大gap，并将最大gap位置及之后所有位置的
        client的is_normal属性改为false，更新normal_client_dict_list。
        返回更新前的方差、更新后的方差以及方差比。
        """
        # 记录更新前 is_normal=True 的 client 的 mse_med 值
        before_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            # 找到对应的 client
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    before_mse_values.append(item["mse_med"])
                    break
        
        # 计算更新前的方差
        if len(before_mse_values) > 1:
            before_mse_tensor = torch.stack(before_mse_values)
            before_variance = torch.var(before_mse_tensor, unbiased=False)
        elif len(before_mse_values) == 1:
            before_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            before_variance = torch.tensor(0.0, dtype=torch.float32)
        
        # 按 mse_med 值进行排序（从小到大）
        sorted_list = sorted(mse_med_list, key=lambda x: x["mse_med"].item())
        
        # 提取排序后的 mse_med 值用于计算差分
        mse_values = [item["mse_med"] for item in sorted_list]
        
        # 计算一阶差分
        if len(mse_values) < 2:
            # 如果只有一个或零个元素，无法计算差分
            # 返回更新前的方差（作为前后方差）和方差比1.0
            return before_variance, before_variance, torch.tensor(1.0, dtype=torch.float32)
        
        # 将 tensor 转换为列表以便计算差分
        mse_values_list = [val.item() for val in mse_values]
        diffs = [mse_values_list[i+1] - mse_values_list[i] for i in range(len(mse_values_list) - 1)]
        
        # 找到最大gap的位置
        max_gap_idx = diffs.index(max(diffs))
        
        # 记录最大gap位置的client id以及之后所有位置的client id
        abnormal_client_ids = []
        # 最大gap在位置i意味着从i到i+1的gap最大，所以应该标记从i+1开始的所有client
        for i in range(max_gap_idx + 1, len(sorted_list)):
            abnormal_client_ids.append(sorted_list[i]["id"])
        
        # 根据这些id将对应client的is_normal属性改为false
        for client in self.clients:
            if client.id in abnormal_client_ids:
                client.is_normal = False
        
        # 更新self.normal_client_dict_list
        self.init_normal_client_ids()
        
        # 记录更新后 is_normal=True 的 client 的 mse_med 值
        after_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            # 找到对应的 client
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    after_mse_values.append(item["mse_med"])
                    break
        
        # 计算更新后的方差
        if len(after_mse_values) > 1:
            after_mse_tensor = torch.stack(after_mse_values)
            after_variance = torch.var(after_mse_tensor, unbiased=False)
        elif len(after_mse_values) == 1:
            after_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            after_variance = torch.tensor(0.0, dtype=torch.float32)
        
        # 计算方差比（后者与前者之比）
        if before_variance.item() != 0:
            variance_ratio = after_variance / before_variance
        else:
            # 如果更新前方差为0，则方差比设为0（或根据业务逻辑设为其他值）
            variance_ratio = torch.tensor(0.0, dtype=torch.float32) if after_variance.item() == 0 else torch.tensor(float('inf'), dtype=torch.float32)
        
        return before_variance, after_variance, variance_ratio

    def _save_client_states(self) -> dict:
        """保存所有client的is_normal状态"""
        states = {}
        for client in self.clients:
            states[client.id] = client.is_normal
        return states
    
    def _restore_client_states(self, states: dict) -> None:
        """恢复所有client的is_normal状态"""
        for client in self.clients:
            if client.id in states:
                client.is_normal = states[client.id]
        self.init_normal_client_ids()

    def iterative_update_normal_clients(self, gradient_attack_func: callable = None) -> dict:
        """
        迭代更新normal clients，记录最低方差比的时刻，并将该时刻的状态作为最终状态。
        停止条件：当normal client数量小于总client数量的1/2时停止。
        返回包含最终状态信息的字典。

        参数:
            gradient_attack_func: 梯度层面的攻击函数，在筛选阶段的本地训练中应用
        """
        # 确保从所有client.is_normal均为true开始
        for client in self.clients:
            client.is_normal = True
        self.init_normal_client_ids()
        
        total_clients = len(self.clients)
        min_variance_ratio = float('inf')
        best_state = None
        best_normal_client_dict_list = None
        iteration = 0
        
        # 记录每次迭代的信息
        iteration_records = []
        
        while True:
            # 检查停止条件：normal client数量是否小于总client数量的1/2
            normal_count = sum(1 for client in self.clients if client.is_normal)
            if normal_count < total_clients / 2:
                break

            # 收集当前迭代的mse_med
            mse_med_list = self.collect_client_mse_med()

            # 如果normal client数量为0，无法继续
            if normal_count == 0:
                break

            # 执行更新
            before_var, after_var, variance_ratio = self.update_normal_clients_by_gap(mse_med_list)
            variance_ratio_value = variance_ratio.item()
            print(f"  Variance ratio: {variance_ratio_value:.6f} (before:{before_var:.6f}, after:{after_var:.6f})")
            
            # 保存更新后的状态
            updated_state = self._save_client_states()
            updated_normal_client_dict_list = [dict(d) for d in self.normal_client_dict_list]
            
            # 记录本次迭代信息
            iteration_records.append({
                "iteration": iteration,
                "variance_ratio": variance_ratio_value,
                "normal_count": normal_count,
                "state": updated_state.copy(),
                "normal_client_dict_list": [dict(d) for d in updated_normal_client_dict_list]
            })
            
            # 检查是否找到更低的方差比（更新后的状态）
            if variance_ratio_value < min_variance_ratio:
                min_variance_ratio = variance_ratio_value
                best_state = updated_state.copy()
                best_normal_client_dict_list = [dict(d) for d in updated_normal_client_dict_list]
            
            iteration += 1
            
            # 检查更新后是否还能继续（防止无限循环）
            new_normal_count = sum(1 for client in self.clients if client.is_normal)
            if new_normal_count == normal_count:
                # 如果没有client被标记为abnormal，说明无法继续更新
                break
        
        # 恢复到最低方差比时刻的状态
        if best_state is not None:
            self._restore_client_states(best_state)
            self.normal_client_dict_list = best_normal_client_dict_list
        
        return {
            "min_variance_ratio": min_variance_ratio,
            "final_normal_client_dict_list": self.normal_client_dict_list,
            "total_iterations": iteration,
            "iteration_records": iteration_records
        }

    def train(self, epochs:int, learning_rate:float, aggregation:callable, save_path:str = None,
              gradient_attack_func:callable = None) -> None:
        """
        训练流程：
        1. 调用iterative_update_normal_clients确定最终的normal_client_dict_list
        2. 进入epochs次循环，每次循环：
           - 将当前的model发送到每一个正常的client上
           - 通过client.train()收集各个正常client的梯度
           - 如果有gradient_attack_func，则对梯度应用攻击
           - 将这些梯度打包为列表作为aggregation的传入参数
           - 利用返回值（整合后的梯度）对当前模型的参数进行更新

        参数:
            gradient_attack_func: 梯度层面的攻击函数，接收gradients_list返回攻击后的gradients_list
        """

        # 1. 确定最终的normal_client_dict_list（筛选阶段，包含本地训练）
        self.iterative_update_normal_clients(gradient_attack_func)

        # 打印被剔除的异常 Client ID
        abnormal_ids = sorted([client.id for client in self.clients if not client.is_normal])
        normal_count = len(self.clients) - len(abnormal_ids)
        print(f"\n[Screening Result] Identified {len(abnormal_ids)} abnormal clients.")
        if len(abnormal_ids) > 0:
            print(f"   -> Abnormal Client IDs: {abnormal_ids}")
        print(f"[Screening Result] {normal_count} normal clients will participate in training.\n")

        # 2. 进入循环梯度更新
        # 使用筛选阶段确定的正常客户端ID列表
        normal_client_ids = [d['id'] for d in self.normal_client_dict_list]

        for epoch in range(epochs):
            # 2.1 将当前模型发送到正常客户端
            for client in self.clients:
                if client.id in normal_client_ids:
                    client.model = copy.deepcopy(self.model)

            # 2.2 收集正常客户端的梯度
            gradients_list = []
            for client in self.clients:
                if client.id in normal_client_ids:
                    gradient_result = client.train(local_epochs=1, learning_rate=learning_rate)
                    # 带上攻击者标签，以便攻击函数识别漏网之鱼
                    gradient_result['is_attacker'] = client.is_attacker
                    gradients_list.append(gradient_result)

            # 计算聚合前的平均loss用于监控 (这其实是本地训练动态Loss)
            local_train_loss = sum([g['loss'] for g in gradients_list]) / len(gradients_list)

            # 让漏网的坏节点实施攻击！
            if gradient_attack_func is not None:
                gradients_list = gradient_attack_func(gradients_list)

            # 2.3 聚合梯度
            aggregated_gradients = aggregation(gradients_list)

            # 2.4 利用整合后的梯度对当前模型的参数进行更新
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in aggregated_gradients:
                        # 假设aggregation返回的是梯度更新量，直接加到参数上
                        # 如果需要使用学习率等，可以在aggregation函数中处理
                        param.data -= 1.0 * aggregated_gradients[name]

            # 2.6 计算聚合后的loss（用更新后的模型）
            self.model.eval()
            total_loss_after = 0.0
            total_samples = 0
            with torch.no_grad():
                for client in self.clients:
                    if client.id in normal_client_ids:
                        for batch_data in client.data:
                            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                                inputs, labels = batch_data
                            else:
                                continue
                            device = next(self.model.parameters()).device
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = self.model(inputs)
                            loss = F.cross_entropy(outputs, labels)
                            total_loss_after += loss.item() * inputs.size(0)
                            total_samples += inputs.size(0)
            avg_loss_after = total_loss_after / total_samples if total_samples > 0 else 0
            self.model.train()

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Local Train Loss (Avg): {local_train_loss:.6f} | "
                  f"Global Aggregated Model Loss: {avg_loss_after:.6f}")

        # 3. 保存模型参数
        if save_path is None:
            # 如果未传入路径，优先使用模型的 save_path 属性
            if hasattr(self.model, 'save_path') and self.model.save_path:
                save_path = self.model.save_path
            else:
                # 如果都没有，使用默认文件名
                save_path = "model_params.pth"
        
        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")