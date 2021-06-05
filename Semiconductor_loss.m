function [P_cond_sw_inner,P_cond_sw_outer,P_cond_dio_clp,P_cond_dio_addi,...
        P_cond_dio_rec] = Semiconductor_loss(I_sw_inner_rms_actual,...
        I_sw_outer_rms_actual,I_dio_clp_ave_actual,I_dio_addi_ave_actual,...
        I_dio_rec_ave_ideal,R_on,V_d_pri,V_d_rec)
%This function calculates the conduction losses from the semiconductors
%
    N_inner_sw = 2;
    N_outer_sw = 2;
    N_clp_dio = 2;
    N_addi_dio = 2;
    N_rec_dio = 4;
    
    % conduction loss from inner switches
    P_cond_sw_inner = N_inner_sw * I_sw_inner_rms_actual ^ 2 * R_on;
    
    % conduction loss from outer switches
    P_cond_sw_outer = N_outer_sw * I_sw_outer_rms_actual ^ 2 * R_on;
    
    % conduction loss from clamping diodes
    P_cond_dio_clp = N_clp_dio * I_dio_clp_ave_actual * V_d_pri;
    
    % conduction loss from additional diodes
    P_cond_dio_addi = N_addi_dio * I_dio_addi_ave_actual * V_d_pri;
    
    % conduction loss from rectifier
    P_cond_dio_rec = N_rec_dio * I_dio_rec_ave_ideal * V_d_rec;
end

