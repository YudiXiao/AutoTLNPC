function [P_winding_Trp,P_winding_Trs,P_core_Tr] = transformer_loss(...
                    I_tr_p_rms_actual,I_tr_s_rms_actual,V_trp_PS,...
                    V_trp_delta_t1,V_trp_delta_t2,R_T_p,R_T_s,f_s,...
                    PS,delta_t1,delta_t2,core_k,core_a,core_b,N_p,N_s,Ae,Ve)
% The function transformer_loss calculates the winding loss and core loss 
% of the transformer.

    % winding loss
    P_winding_Trp = I_tr_p_rms_actual ^ 2 * R_T_p * N_p;
    P_winding_Trs = I_tr_s_rms_actual ^ 2 * R_T_s * N_s;
    
    % core loss
    
    F_sample = 250e6; % sampling frequency
    T_sample = 1 / F_sample; % sampling period
    T_s = 1 / f_s; % switching cycle
    L_signal = round(2 * (PS * T_s / 2 + delta_t1 + delta_t2) / T_sample); % length of signal
    t = (0:L_signal-1) * T_sample;  % Time vector
    
    v_signal = zeros(1,L_signal);
    v_signal(1,1:round((PS * T_s / 2) / T_sample)) = V_trp_PS;
    v_signal(1,round((PS * T_s / 2) / T_sample) + 1:round((PS * T_s / 2 + delta_t1)...
                / T_sample)) = V_trp_delta_t1;    
    v_signal(1,round((PS * T_s / 2 + delta_t1)/ T_sample) + 1:round((PS * T_s / 2 + ...
                    delta_t1 + delta_t2) / T_sample)) = V_trp_delta_t2;
    for i = round((PS * T_s / 2 + delta_t1 + delta_t2) / T_sample) + 1:1:L_signal
        v_signal(1,i) = - v_signal(1,i-round((PS * T_s / 2 + delta_t1 + delta_t2)...
                        / T_sample));
    end
    
%     figure
%     plot(t,v_signal);

    fft_v_signal = fft(v_signal);
    P2 = abs(fft_v_signal / L_signal);
    P1 = P2(1:L_signal / 2 + 1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    f_fft = F_sample * (0:(L_signal / 2)) / L_signal; % in Hz
%     plot(f_fft,P1) 
%     title('Single-Sided Amplitude Spectrum of signal')
%     xlabel('f (Hz)')
%     ylabel('|P1(f)|')

    B_peak = (P1 / (N_p * Ae)) ./ (2 * pi * f_fft); % peak flux density, in T
    
    P_core_v = core_k * ((f_fft / 1000) .^ core_a) .* (B_peak .^ core_b); % k in kW/m^3
    P_core_v(isnan(P_core_v)) = 0;
    P_core_v_total = sum(P_core_v,'all'); % power loss per volume, kW/m^3
    
    % core loss in W
    P_core_Tr = 1000 * P_core_v_total * Ve;

end

