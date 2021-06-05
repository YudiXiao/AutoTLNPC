function [RT_inner,RT_outer] = Switching_performance(C_oss,ratio_div_sw_actual,I_3,V_in)
% The function estimate the switching performance of the converter.
% The switching performance is estimated by calculating the time for
% charging / discharging the Coss. The longer the time, the harder it is to
% achieve zero-voltage switching
%   
    
    RT_inner = C_oss * (V_in / 2) / (I_3 / 2);
    RT_outer = C_oss * (V_in / 2) / (I_3 * ratio_div_sw_actual / 2);
end

