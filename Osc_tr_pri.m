function [Osc_pri_peak] = Osc_tr_pri(C_p,L_kp,L_ks,L_ac_1,L_ac_2,N_p,N_s,...
                        V_in,rise_time,f_s,R_Lac_1,R_Lac_2)
% The function Osc_tr_pri calculates the voltage oscillation on the
% transformer primary side. This oscillation happens when the switching
% node has a voltage swing while the secondary side of the transformer is
% shorted by the rectifier.
    
    % total leakage inductance referred to primary side
    L_k = L_kp + (N_p / N_s) ^ 2 * L_ks;
    
    L_ac = L_ac_1 + L_ac_2;
    R_Lac = R_Lac_1 + R_Lac_2;
    
    sys = tf([L_k 0],[L_ac*L_k*C_p L_k*C_p*R_Lac L_k+L_ac R_Lac]);
    
    F_sample = 250e6; % sampling frequency
    T_sample = 1 / F_sample; % sampling period
    T_s = 1 / f_s; % switching cycle
    
    t = 0:T_sample:T_s/2;
    
    slew_rate = (V_in / 2) / rise_time;
    
    u = zeros(1,size(t,2));
    
    for i=1:1:size(t,2)
        if t(1,i) <= rise_time
            u(1,i) = slew_rate * t(1,i);
        else
            u(1,i) = V_in / 2;
        end
    end
    
%     lsim(sys,u,t)
%     grid on
    y = lsim(sys,u,t);
    
    Osc_pri_peak = max(y);
end

