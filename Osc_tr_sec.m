function [P_snubber,R_snubber] = Osc_tr_sec(C_p,C_s,L_kp,L_ks,...
                                        L_ac_1,L_ac_2,N_p,N_s,V_in,...
                                        R_Lac_1,R_Lac_2,R_lkp,...
                                        R_lks,Lac_s,C_r,I_2,f_s,...
                                        V_o,V_cp)
% The function Osc_tr_sec calculates the oscillations on both primary
% and secondary side of the transformer. This oscillation happens when
% the recitifier on the secondary side stops operating in free-wheeling
% mode.
%
%The function also calculates the loss in snubber.

    T_s = 1 / f_s; % switching cycle
    
    set_param('Osc_sec/DC Voltage Source','Amplitude',num2str(V_in/2));
    set_param('Osc_sec/Series RLC Branch','Capacitance',num2str(C_p));
    set_param('Osc_sec/Series RLC Branch1','Capacitance',num2str(C_s/((N_p/N_s)*(N_p/N_s))));
    set_param('Osc_sec/Series RLC Branch2','Resistance',num2str(R_Lac_2),...
        'Inductance',num2str(L_ac_2));
    set_param('Osc_sec/Series RLC Branch3','Resistance',num2str(R_lkp),...
        'Inductance',num2str(L_kp));
    set_param('Osc_sec/Series RLC Branch4','Resistance',num2str(R_Lac_1),...
        'Inductance',num2str(L_ac_1),'InitialCurrent',num2str(I_2));
    set_param('Osc_sec/Series RLC Branch5','Resistance',num2str(R_lks*(N_p/N_s)*(N_p/N_s)),...
        'Inductance',num2str(L_ks*(N_p/N_s)*(N_p/N_s)));
    set_param('Osc_sec/Series RLC Branch6','Inductance',num2str(Lac_s*(N_p/N_s)*(N_p/N_s)));
    set_param('Osc_sec/Series RLC Branch7','Capacitance',num2str(C_r/((N_p/N_s)*(N_p/N_s))));
     set_param('Osc_sec','StopTime',num2str(1/f_s))
    
    Osc_sec_output = sim('Osc_sec');
    Osc_sec_v = (N_s / N_p) * (Osc_sec_output.ts_Osc_sec.data);
    
    V_osc_sec_peak = max(Osc_sec_v);
    
% The equations below are derived from 'Analysis and design 
% for RCD clamped snubber used in output rectifier of phase 
% shift full-bridge ZVS converters'

    R_snubber = T_s * (V_cp - V_o) * (V_cp - V_in / 2) / ...
                (C_r * V_cp * (V_osc_sec_peak - V_cp));
    P_snubber = (V_cp - V_o) ^ 2 / R_snubber;
    
end

