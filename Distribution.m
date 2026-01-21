function [clients] = Distribution(train_data, train_labels, K)
    % 这里也可以使用load来读取mat文件
    
    % 按标签分组数据（0-9）
    label_groups = cell(10, 1);
    for label = 0:9
        label_indices = find(train_labels == label);
        label_groups{label+1} = label_indices;
        fprintf('数字 %d 的图片数量: %d\n', label, length(label_indices));
    end
    
    % 创建客户端数据结构
    clients = struct();
    for i = 1:K
        clients(i).id = i;
        clients(i).data = [];
        clients(i).labels = [];
    end
    
    % Non-IID分配策略：每个客户端偏向某些数字
    % 为每个客户端分配主要标签和次要标签（允许多个客户端有相同的主要标签）
    primary_labels = randi(10, 1, K); % 随机生成1-10之间的数字，长度为K
    secondary_labels = mod(primary_labels + randi([1, 9], 1, K) - 1, 10) + 1; % 每个客户端一个次要标签
    
    fprintf('\n各客户端的主要和次要标签:\n');
    for i = 1:K
        fprintf('客户端 %d: 主要标签=%d, 次要标签=%d\n', i, primary_labels(i)-1, secondary_labels(i)-1);
    end
    
    % 分配数据到客户端
    for label = 0:9
        indices = label_groups{label+1};
        num_images_label = length(indices);
        
        % 计算每个客户端应该获得的该标签图片数量
        % 主要标签客户端获得更多该标签图片，次要标签客户端获得较少，其他客户端获得很少
        label_distribution = zeros(K, 1);
        
        for i = 1:K
            if primary_labels(i) == label+1
                % 主要标签客户端获得40%的该标签图片
                label_distribution(i) = round(0.4 * num_images_label);
            elseif secondary_labels(i) == label+1
                % 次要标签客户端获得30%的该标签图片
                label_distribution(i) = round(0.3 * num_images_label);
            else
                % 其他客户端均分剩余的30%图片
                label_distribution(i) = round(0.3 * num_images_label / (K-2));
            end
        end
        
        % 调整分配数量以确保总和不超过可用图片数量
        total_allocated = sum(label_distribution);
        if total_allocated > num_images_label
            % 按比例减少分配
            scale_factor = num_images_label / total_allocated;
            label_distribution = floor(label_distribution * scale_factor);
            
            % 处理可能的整数截断导致的少量剩余图片
            remaining = num_images_label - sum(label_distribution);
            if remaining > 0
                % 将剩余图片分配给主要标签客户端
                [~, idx] = max(label_distribution);
                label_distribution(idx) = label_distribution(idx) + remaining;
            end
        end
        
        % 随机打乱当前标签的图片索引
        shuffled_indices = indices(randperm(num_images_label));
        
        % 分配图片给客户端
        start_idx = 1;
        for i = 1:K
            num_to_assign = label_distribution(i);
            if num_to_assign > 0
                end_idx = start_idx + num_to_assign - 1;
                if end_idx > num_images_label
                    end_idx = num_images_label;
                    num_to_assign = end_idx - start_idx + 1;
                end
                
                assigned_indices = shuffled_indices(start_idx:end_idx);
                clients(i).data = [clients(i).data; train_data(assigned_indices, :)];  % 二维矩阵
                clients(i).labels = [clients(i).labels; train_labels(assigned_indices)];  % 列向量
                
                start_idx = start_idx + num_to_assign;
            end
        end
    end
    
    % 计算每个客户端的数据量
    for i = 1:K
        clients(i).num_images = size(clients(i).data, 1);
    end
    
    % 显示分配结果
    fprintf('\n=== Non-IID 分配结果 ===\n');
    for i = 1:K
        fprintf('客户端 %d: %d 张图片\n', i, clients(i).num_images);
        
        % 显示该客户端的标签分布
        label_counts = zeros(10, 1);
        for j = 0:9
            label_counts(j+1) = sum(clients(i).labels == j);
        end
        fprintf('  标签分布: %s\n', mat2str(label_counts'));
    end
    
    % 可视化每个客户端的标签分布
    figure('Name', '各客户端Non-IID标签分布');
    for i = 1:min(K, 9) % 最多显示9个客户端
        subplot(3, 3, i);
        
        % 计算标签分布
        label_counts = zeros(10, 1);
        for j = 0:9
            label_counts(j+1) = sum(clients(i).labels == j);
        end
        
        bar(0:9, label_counts);
        title(sprintf('客户端 %d (共%d张)', i, clients(i).num_images));
        xlabel('数字');
        ylabel('数量');
        xlim([-0.5, 9.5]);
        grid on;
    end
    
    % 可视化整体分配情况
    figure('Name', '客户端数据量分布');
    client_nums = [clients.num_images];
    bar(1:K, client_nums);
    xlabel('客户端 ID');
    ylabel('图片数量');
    title(sprintf('各客户端数据量分布 (总计: %d)', sum(client_nums)));
    grid on;
    
    % 保存分配结果
    save('non_iid_client_data.mat', 'clients', 'K');
    
    fprintf('\nNon-IID数据分配完成！\n');
end

