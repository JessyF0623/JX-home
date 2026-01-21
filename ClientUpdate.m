function [u_k, avg_error, num_epochs_used] = ClientUpdate(inputNums, hideNums, outputNums, E, B, eta, sigma, in_k, expectout_k, z_init, k, lambda, beta)
    % 提取v和w
    v_size = inputNums * hideNums;
    w_size = hideNums * outputNums;

    v_flat = z_init(1:v_size);
    v = reshape(v_flat, inputNums, hideNums);

    w_flat = z_init(v_size + 1 : v_size + w_size);
    w = reshape(w_flat, hideNums, outputNums);
    
    % 获取样本总数
    total_samples = size(in_k, 1);  % 行数，获取当前第k个client的图片数量
    
    % 计算批次数（向下取整）
    num_batches = floor(total_samples / B);
    
    error_last = compute_error(in_k, expectout_k, v, w, lambda(:, k), ...
        beta, z_init, inputNums, outputNums); % 初始误差，指的是当前round开始之前的error
    error_norm_last = norm(error_last);
    
    epoch = 0;
    epoch_error = 0;
    converged = false;

    % 训练循环
    while epoch < E && ~converged
    % for epoch = 1:E
        epoch = epoch + 1;

        % 按批次处理
        for batch = 1:num_batches   % 接下来就是对每一张图片进行训练
            % 计算当前批次的样本范围
            start_idx = (batch-1)*B + 1;
            end_idx = min(batch*B, total_samples);
            
            % 获取输入和期望输出
            x_batch = in_k(start_idx:end_idx, 1:inputNums);  % 修改，并行运算

            % 获取当前批次的标签向量
            labels_batch = expectout_k(start_idx:end_idx);

            % 初始化one-hot编码矩阵
            d_batch = zeros(length(labels_batch), outputNums);  % 大小正确
            
            % 生成one-hot编码
            for i = 1:length(labels_batch)
                class_index = labels_batch(i) + 1; % 因为标签从0开始，Matlab索引从1开始
                d_batch(i, class_index) = 1;
            end
            
            % 前向传播
            [y1, output_batch] = forwardPropagation(x_batch, v, w);  % output_batch大小为B * outputNums
            
            % 反向传播 + 权重更新
            [v, w, error_current] = backPropagation(x_batch, d_batch, y1, output_batch, ...
                    v, w, eta, lambda, beta, k, z_init,...
                    inputNums, hideNums, outputNums);

            % 计算当前误差范数
            error_norm_current = norm(error_current);
            if error_norm_current <= sigma * error_norm_last
                break; % 满足不精确准则，提前退出
            end

            % 计算交叉熵损失
            batch_error = -sum(d_batch .* log(output_batch + 1e-9), 'all'); % 添加1e-9避免log(0)
            epoch_error = epoch_error + batch_error;
        end

        if error_norm_current <= sigma * error_norm_last
            converged = true;
            break;
        end
        % error_norm_last = error_current;   % 注释！！！！初始误差不是每次都更新，只在梯度下降前更新一次
    end
    
    avg_error = epoch_error / total_samples;
    num_epochs_used = epoch;

    % 输出最终权值
    u_k = [v(:); w(:)];
end
