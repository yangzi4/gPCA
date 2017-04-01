rng(1);
val_mat = zeros(4, 11) - 1;

for alpha = 0:10:100
    col_ = alpha/10 + 1;
    for D_t = 1:4
        val = 0;
        for gen_seed = 0:24
            str1 = strcat('data(13x16)x0_a', num2str(alpha), ...
                'D', num2str(D_t), 's20_', num2str(gen_seed), '.txt');
            disp(str1)
            data_1 = importdata(str1);
            str2 = strcat('data(13x16)x1_a', num2str(alpha), ...
                'D', num2str(D_t), 's20_', num2str(gen_seed), '.txt');
            data_2 = importdata(str2);
            str3 = strcat('data(13x16)x2_a', num2str(alpha), ...
                'D', num2str(D_t), 's20_', num2str(gen_seed), '.txt');
            data_3 = importdata(str3);
            data_mat = {data_1, data_2, data_3};
            rng(1)
            [J_, A_, r_, rk_] = JIVE_RankSelect(data_mat, 0.05, 100);
            %disp(r_)
            %disp(rk_)
            if r_ == D_t
                val = val + 1;
            end
        val_mat(D_t, col_) = val;
        end
    save('JIVEaccu_(100x400)x2_5-10_25.txt', 'val_mat', '-ascii');
    end
end

disp(val_mat)
save('JIVEaccu_(13x16)x3_D4s20-25.txt', 'val_mat', '-ascii')