function [client_data, all_sample_counts]= GenerateClientData(K)
% 生成 K 个客户端的不同输入数据

    % 参数设置
    min_samples = 50;   % 最小样本数量
    max_samples = 450;  % 最大样本数量
    
    % 确保每个客户端样本数量不同
    all_sample_counts = randperm(max_samples - min_samples + 1, K) + min_samples - 1;
    
    % 初始化存储客户端数据的元胞数组
    client_data = cell(1, K);
    
    for k = 1:K
        % 当前客户端的样本数量
        n_samples = all_sample_counts(k);
        
        % 计算所需区间长度 (基于样本数量和步长)
        range_length = (n_samples - 1) * 0.1;
        
        % 随机生成左端点 (确保右端点不超过+20)
        max_left = 20 - range_length;
        left_end = -20 + (max_left + 20) * rand();
        
        % 计算右端点
        right_end = left_end + range_length;
        
        % 生成输入数据 (使用 linspace 确保精确样本数量)
        client_data{k} = linspace(left_end, right_end, n_samples);
        
        % 显示客户端信息
        fprintf('客户端 %d: 样本数 = %d, 范围 = [%.2f, %.2f]\n', ...
                k, n_samples, left_end, right_end);
    end
end