% 反向传播函数
function [v, w, error] = backPropagation(x, d, y1, output, v, w, eta, lambda, beta, k, z_init, inputNums, hideNums, outputNums)
    % 对于Softmax + 交叉熵损失，输出层误差简化为 (output - d)
    deltk = output - d;
    
    % 计算隐层误差
    deltj = y1 .* (1 - y1) .* (deltk * w');
    
    % 更新权重
    deltw = y1' * deltk;
    deltv = x' * deltj;
    deltu = [deltv(:); deltw(:)];
    u = [v(:); w(:)];

    % 定义误差项error，最优时error=0
    error = deltu - lambda(:, k) + beta * (u - z_init);
    u = u - eta * error;

    % u = u - eta * deltu;
    
    % 提取v和w
    v_size = inputNums * hideNums;
    w_size = hideNums * outputNums;
    
    v_flat = u(1:v_size);
    v = reshape(v_flat, inputNums, hideNums);
    
    w_flat = u(v_size + 1 : v_size + w_size);
    w = reshape(w_flat, hideNums, outputNums);
    % v = v - eta * deltv;
    % w = w - eta * deltw;
end

