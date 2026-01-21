% 前向传播函数
function [y1, output] = forwardPropagation(x, v, w)
    % 隐层计算
    out1 = x * v;
    y1 = 1./(1 + exp(-out1));  % Sigmoid激活函数
    
    % 输出层计算
    out2 = y1 * w;
    
    % 应用Softmax函数
    % 减去最大值以提高数值稳定性
    max_out2 = max(out2, [], 2);
    exp_out2 = exp(out2 - max_out2);
    sum_exp = sum(exp_out2, 2);
    output = exp_out2 ./ sum_exp;
end
