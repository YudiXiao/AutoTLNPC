function [I_ac_p_rms,I_sw_inner_rms_actual,I_sw_outer_rms_actual,...
            I_dio_clp_ave_actual,I_dio_addi_ave_actual,I_tr_p_rms_actual,...
            I_tr_s_rms_actual,I_dio_rec_ave_ideal,I_Lf_rms,V_Lac_PS,...
            V_Lac_delta_t1,V_Lac_delta_t2,V_trp_PS,V_trp_delta_t1,...
            V_trp_delta_t2,V_Lf_PS_delta_t1,V_Lf_delta_t2,delta_t1,...
            delta_t2,I_3,I_2,D_loss,I_Lf_ripple,ratio_div_sw_actual] = circuit_state(V_in,V_o,I_o,L_f,N_p,...
                                        N_s,L_ac_1,L_ac_2,f_s,PS)
% The function circuit_state calculates the state of the circuits.
% The calculated circuit states are used for evaluating the circuit's
% performance.
%

    T_s = 1 / f_s; % switching cycle

    L_ac = L_ac_1 + L_ac_2;
    
% calculate duty ratio loss: D_loss
    D_loss = (8 * L_ac * I_o * N_s) / (T_s * V_in * N_p);
    delta_t1 = D_loss * T_s / 2;    % duty ratio loss in time
    delta_t2 = T_s / 2 - PS * T_s / 2 - delta_t1;
    
% some constants used in later calculations
    I_rate1 = V_in / (2 * L_ac);    % rate of change of primary ac current 
                                    % during duty loss
    I_rate2 = (V_in / 2 - N_p * V_o / N_s) / (L_ac + (N_p / N_s) ^ 2 ...
                * L_f); % rate of change of primary ac current
                        % during magnetizing L_f
    I_3 = 0.5 * ((I_rate1 - I_rate2) * delta_t1 + T_s * (1 - PS) * ...
                I_rate2 / 2);   % maximum primary ac current
    I_1 = -I_3; % minimum primary ac current
    I_2 = I_1 + I_rate1 * delta_t1; % instant value of primary ac current 
                % at the end of the duty loss
%
    Coss_vs_Vds = csvread('Coss_vs_Vds.csv');
    Cd_vs_Vr = csvread('Cd_vs_Vr.csv');
    t_stp = 100e-12; % time step 0.1 ns
    t_total = 38e-9; % total time span 100 ns
    t_array = 0:t_stp:t_total;
    data_length = size(t_array,2);
    
    i_ac = zeros(data_length,1);
    v_ds_1 = zeros(data_length,1);
    v_ds_4 = zeros(data_length,1);
    v_r = zeros(data_length,1);
    
    V_ds_1_ini = V_in / 2;
    V_ds_4_ini = 0;
    V_r_ini = (V_in * N_s) / (2 * N_p);
    
    for i = 1:1:data_length
        
        if i == 1
            i_ac(i,1) = I_3;
            v_ds_1(i,1) = V_ds_1_ini;
            v_ds_4(i,1) = V_ds_4_ini;
            v_r(i,1) = V_r_ini;
        else
            [val,idx]=min(abs(Coss_vs_Vds(:,1) - v_ds_1(i - 1,1)));
            Coss1 = Coss_vs_Vds(idx,2);
            [val,idx]=min(abs(Coss_vs_Vds(:,1) - v_ds_4(i - 1,1)));
            Coss4 = Coss_vs_Vds(idx,2);
            [val,idx]=min(abs(Cd_vs_Vr(:,1) - v_r(i - 1,1)));
            Cd = Cd_vs_Vr(idx,2);
            
            i_ac(i,1) = I_3 * (Coss1 + Coss4) / (Coss1 + Coss4 + 2 * (N_s / N_p) ^ 2 * Cd);
        
            v_ds_1(i,1) = v_ds_1(i - 1,1) - abs(i_ac(i - 1,1)) * t_stp * (1e12) / (Coss1 + Coss4);

            v_ds_4(i,1) = v_ds_4(i - 1,1) + abs(i_ac(i - 1,1)) * t_stp * (1e12) / (Coss1 + Coss4);

            v_r(i,1) = v_r(i - 1,1) - abs(I_3 - i_ac(i - 1,1)) * t_stp * N_p * (1e12) / (2 * N_s * Cd);
        end
        
    end
    
    ratio_div_sw_ideal = 0.5;
    ratio_div_clp_ideal = 0.5;
    
    ratio_div_sw_actual = i_ac(end) / i_ac(1);
    ratio_div_clp_actual = ratio_div_sw_actual; 
    ratio_div_addi_actual = 1 - ratio_div_sw_actual;
% calculate RMS value of primary ac current (inductor current): I_ac_p_rms
    I_ac_2_t_1 = I_1 * I_1 * PS * T_s / 2;
    I_ac_2_t_2 = (I_1 - I_rate1 * PS * T_s / 2) ^ 2 * delta_t1 + (I_1 - ...
                    I_rate1 * PS * T_s / 2) * I_rate1 * ((PS * T_s / 2 + ...
                    delta_t1) ^ 2 - (PS * T_s / 2) ^ 2) + I_rate1 ^ 2 * ...
                    ((PS * T_s / 2 + delta_t1) ^ 3 - (PS * T_s / 2) ^ 3) / 3;
    I_ac_2_t_3 = (I_2 - I_rate2 * (PS * T_s / 2 + delta_t1)) ^ 2 * delta_t2 + ...
                    (I_2 - I_rate2 * (PS * T_s / 2 + delta_t1)) * I_rate2 * ...
                    ((T_s / 2) ^ 2 - (PS * T_s / 2 + delta_t1) ^ 2) + ...
                    I_rate2 ^ 2 * ((T_s / 2) ^ 3 - (PS * T_s / 2 + ...
                    delta_t1) ^ 3) / 3;
    I_ac_p_rms = sqrt((I_ac_2_t_1 + I_ac_2_t_2 + I_ac_2_t_3) / (T_s / 2));
    
% calculate RMS value of switch current: I_sw_rms
    % In ideal case
    I_sw_inner_rms_ideal = I_ac_p_rms / sqrt(2);
    I_sw_outer_2_t1_ideal = (I_1 * ratio_div_sw_ideal) ^ 2 * PS * T_s / 2;
    I_sw_outer_rms_ideal = sqrt((I_sw_outer_2_t1_ideal + I_ac_2_t_2 + ...
                            I_ac_2_t_3) / T_s);
                        
    % In actual case
     I_sw_inner_rms_actual = I_ac_p_rms / sqrt(2);
     I_sw_outer_2_t1_actual = (I_1 * ratio_div_sw_actual) ^ 2 * PS * T_s / 2;
     I_sw_outer_rms_actual = sqrt((I_sw_outer_2_t1_actual + I_ac_2_t_2 + ...
                            I_ac_2_t_3) / T_s);
    
% calculate average value of primary diode current: I_dio_ave
    % In ideal case
    I_dio_clp_ave_ideal = ((I_3 * ratio_div_clp_ideal) * PS * T_s / 2) / T_s;
    I_dio_addi_ave_ideal = 0;
    
    % In actual case
    I_dio_clp_ave_actual = ((I_3 * ratio_div_clp_actual) * PS * T_s / 2) / T_s;
    I_dio_addi_ave_actual = ((I_3 * ratio_div_addi_actual) * PS * T_s / 2) / T_s;
    
% calculate RMS value of transformer primary current: I_tr_p_rms
    % In ideal case
    I_tr_p_rms_ideal = I_ac_p_rms;
    
    % In actual case
    I_trp_2_t1 = (I_1 * (1 - ratio_div_addi_actual)) ^ 2 * PS * T_s / 2;
    I_trp_2_t2 = I_ac_2_t_2;
    I_trp_2_t3 = I_ac_2_t_3;
    I_tr_p_rms_actual = sqrt((I_trp_2_t1 + I_trp_2_t2 + I_trp_2_t3) / (T_s / 2));

% calculate RMS value of transformer secondary current: I_tr_s_rms
    % In ideal case
    I_tr_s_rms_ideal = I_tr_p_rms_ideal * N_p / N_s;
    
    % In actual case
    I_tr_s_rms_actual = I_tr_p_rms_actual * N_p / N_s;
    
% calculate average value of secondary diode current: I_dio_rec_ave
    % In ideal case
    I_dio_rec_ave_ideal = I_o / 2;
    
% calculate rms value of output inductor current: I_Lf_rms
    %
    I_2_Lf = I_2 * N_p / N_s;
    I_3_Lf = I_3 * N_p / N_s;
    I_Lf_rate_1 = (I_3_Lf - I_2_Lf) / delta_t2;
    I_Lf_rate_2 = (I_2_Lf - I_3_Lf) / (T_s / 2 - delta_t2);
    I_Lf_2_1 = I_2_Lf ^ 2 * delta_t2 + I_2_Lf * I_Lf_rate_1 * delta_t2 ^ 2 + ...
                I_Lf_rate_1 ^ 2 * delta_t2 ^ 3 / 3;
    I_Lf_2_2 = I_3_Lf ^ 2 * (T_s / 2 - delta_t2) + I_3_Lf * I_Lf_rate_2 * ...
                (T_s ^ 2 / 4 - delta_t2 ^ 2) + I_Lf_rate_2 ^ 2 * ...
                (T_s ^ 3 / 8 - delta_t2 ^ 3) / 3;
    I_Lf_rms = sqrt((I_Lf_2_1 + I_Lf_2_2) / (T_s / 2));
    I_Lf_ripple = (I_3_Lf - I_2_Lf) / I_o;
    
% calculate voltage across ac inductor during half switching period
    V_Lac_PS = 0;
    V_Lac_delta_t1 = V_in / 2;
    V_Lac_delta_t2 = I_rate2 * L_ac;
    
% calculate voltage across transformer
    V_trp_PS = 0;
    V_trp_delta_t1 = 0;
    V_trp_delta_t2 = V_in / 2 - V_Lac_delta_t2;
    
% calculate voltage across output inductor
    V_Lf_PS_delta_t1 = - V_o;
    V_Lf_delta_t2 = (I_rate2 * N_p / N_s) * L_f;

end

