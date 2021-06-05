%% Variables
% # switching frequency - 150 to 500 kHz
% # total ac inductance
% # inductance ratio, L_ac_1 / L_ac_2
% # turns ratio
% # snubber voltage
% # filter inductance
% # ferrite material
% # core size for Lac and Tr -  ELP 38/8/25
% # number of stacked cores for transformer
% # number of primary turns of transformer
% # number of stacked cores for ac inductor
% # powder core - number of stacked cores: 2
% # powder core - size fixed to fully use space
% # powder core - MPP material
% # temperature

clc
clear all
close all

% read magnatic component data
%
% read geometric parameters of ELP cores
filename = 'Core_geometry.xlsx';
sheet = 'ELP';
sheet_range = 'C2:S9';
geometry_ELP = xlsread(filename,sheet,sheet_range); % load geometry parameters, mm
geometry_ELP = geometry_ELP./1000;    % convert unit from mm to m
%
% read ferrite material data
filename = 'ferrite_core_loss.xlsx';
ferrite_material = readmatrix(filename,'Range' ,[2 3 17 12]); % ferrite core loss properties
%
% read powder core permeability vs. DC bias data
filename = 'Magnetics-Curve-Fit-Equation-Tool-2020.xlsx';
powder_u_dc_bias = readmatrix(filename,'Range' ,[9 3 18 6]);
%
% read powder core core loss density data
filename = 'Magnetics-Curve-Fit-Equation-Tool-2020.xlsx';
powder_core_loss = readmatrix(filename,'Range' ,[9 11 18 14]);
%
% read powder core permeability vs. frequency data
filename = 'Magnetics-Curve-Fit-Equation-Tool-2020.xlsx';
powder_u_fs = readmatrix(filename,'Range' ,[9 28 18 33]);
%
% read powder core permeability vs. temperature data
filename = 'Magnetics-Curve-Fit-Equation-Tool-2020.xlsx';
powder_u_Ta = readmatrix(filename,'Range' ,[9 38 18 43]);
%

% parameters to sweep

f_s_sweep = 200e3:50e3:250e3;   % Hz

L_ac_sweep = 8e-6:1e-6:10e-6;   % H

L_ac_ratio_sweep = 1:1:1; % L_ac_1 / L_ac_2

T_ratio_sweep = 12/16:2/16:12/16;   % N_p / N_s

V_cp_sweep = 800:20:850;

L_f_sweep = 160e-6:20e-6:160e-6;

ELP_sweep = 5:1:5;

Nstack_Tr_sweep = 3:1:3;

N_p_sweep = 12:2:12;

Nstack_Lac_sweep = 3:1:3;

ferrite_sweep = 4:1:4;

Ldc_material_sweep = 9:1:9;

Ta_sweep = 25:25:25;
%
%
number_of_param = 13;
%
% 1 f_s
% 2 L_ac
% 3 L_ac_ratio
% 4 T_ratio
% 5 V_cp
% 6 L_f
% 7 ELP size
% 8 Nstack_Tr
% 9 N_p
% 10 Nstack_Lac
% 11 ferrite material
% 12 Ldc_material
% 13 Ta
%
number_of_perf = 22;
%
% 1 RT_inner
% 2 RT_outer
% 3 V_gain
% 4 P_cond_sw_innery
% 5 P_cond_sw_outer
% 6 P_cond_dio_clp
% 7 P_cond_dio_addi
% 8 P_cond_dio_rec
% 9 Osc_pri_peak
% 10 P_snubber
% 11 R_snubber
% 12 P_winding_Trp
% 13 P_winding_Trs
% 14 P_core_Tr
% 15 P_winding_Lac
% 16 P_core_Lac
% 17 P_winding_Ldc 
% 18 P_core_Ldc
% 19 D_loss
% 20 I_Lf_ripple
% 21 delta_Lf_percent
% 22 Iacp
%
% data storage
number_of_row = size(f_s_sweep,2) ...
                * size(L_ac_sweep,2) ...
                * size(L_ac_ratio_sweep,2) ...
                * size(T_ratio_sweep,2) ...
                * size(V_cp_sweep,2) ...
                * size(L_f_sweep,2) ...
                * size(ELP_sweep,2)...
                * size(Nstack_Tr_sweep,2) ...
                * size(N_p_sweep,2) ...
                * size(Nstack_Lac_sweep,2) ...
                * size(ferrite_sweep,2) ...
                * size(Ldc_material_sweep,2) ...
                * size(Ta_sweep,2);
number_of_column = number_of_param + number_of_perf;
performance_data = zeros(number_of_row,number_of_column);
Para_ana = zeros(number_of_row,number_of_param);
% Pre-defined variables    

    V_in = 1200;    % input voltage, V
    V_o = 600;  % output voltage, V
    I_o = 5;    % rated output current, A
    PS = 0.1; % phase shift ratio
    
    J = 4e6;  % DC current density, A/m2
    sigma_copper = 5.96e7; % electrical conductivity of copper, S/m
    
    permeability_vacuum = 4*pi*(1e-7);      % H/m
    
    C_oss = 39e-12; % output capacitance of switches, F
    R_on = 160e-3;  % on-state resistance of switches, ohm
    
    V_d_pri = 1.8;  % forward voltage of clamping diodes, V
    
    V_d_rec = 1.8;  % forward voltage of rectifier diodes, V
    
    Lac_s = 100e-9; % secondary ac inductance, H
    
    C_r = 60e-12;   % junction capacitance of the rectifier diode, F
% Start of loop
i = 0;
for para1 = f_s_sweep % switching frequency, Hz
    for para2 = L_ac_sweep
        for para3 = L_ac_ratio_sweep
            for para4 = T_ratio_sweep
                for para5 = V_cp_sweep % snubbering voltage, V
                    for para6 = L_f_sweep % output inductance, H
                        for para7 = ELP_sweep % ELP size
                            for para8 = Nstack_Tr_sweep
                                for para9 =N_p_sweep
                                    for para10 = Nstack_Lac_sweep
                                        for para11 = ferrite_sweep
                                            for para12 = Ldc_material_sweep
                                                for para13 = Ta_sweep
    
                                                    i = i + 1;
                                                    Para_ana(i,1) = para1;
                                                    Para_ana(i,2) = para2;
                                                    Para_ana(i,3) = para3;
                                                    Para_ana(i,4) = para4;
                                                    Para_ana(i,5) = para5;
                                                    Para_ana(i,6) = para6;
                                                    Para_ana(i,7) = para7;
                                                    Para_ana(i,8) = para8;
                                                    Para_ana(i,9) = para9;
                                                    Para_ana(i,10) = para10;
                                                    Para_ana(i,11) = para11;
                                                    Para_ana(i,12) = para12;
                                                    Para_ana(i,13) = para13;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
%%
    Num_worker_ana = 20; % number of worker for parallel computing analytcial models
    
    %set up parallel pool
    load_system('Osc_sec');
    parpool('local',Num_worker_ana);
    
spmd
    % Setup tempdir and cd into it
    currDir = pwd;
    addpath(currDir);
    tmpDir = tempname;
    mkdir(tmpDir);
    cd(tmpDir);
    % Load the model on the worker
    load_system('Osc_sec');
end
    
tic
parfor p = 1:number_of_row
    
    f_s = Para_ana(p,1);
    L_ac = Para_ana(p,2);
    L_ac_ratio = Para_ana(p,3);
    T_ratio = Para_ana(p,4);
    V_cp = Para_ana(p,5);
    L_f = Para_ana(p,6);
    count_ELP = Para_ana(p,7);
    Nstack_Tr = Para_ana(p,8);
    N_p = Para_ana(p,9);
    Nstack_Lac = Para_ana(p,10);
    count_ferrite = Para_ana(p,11);
    Ldc_material = Para_ana(p,12);
    Ta = Para_ana(p,13);
    
    %performance_data(p,:) = Para_ana(p,:);
    
    L_ac_1 = L_ac * L_ac_ratio / (L_ac_ratio + 1); % ac inductance connected to the switching node, H
    L_ac_2 = L_ac * 1 / (L_ac_ratio + 1); % ac inductance connected to the transformer, H
    N_s = N_p / T_ratio;   % number of turns, secondary
    
    rela_permeability_core = ferrite_material(count_ferrite,1);
    
    EE_width_outer = geometry_ELP(count_ELP,10);       % outer width of the core studied, m
    EE_width_inner = geometry_ELP(count_ELP,14);       % inner width of the core studied, m
    EE_width_mid = geometry_ELP(count_ELP,13);            % width of mid leg of the core  studied, m
    EE_length = geometry_ELP(count_ELP,12);
    E_height_outer = geometry_ELP(count_ELP,11);        % outer height of single E core, m
    E_height_inner = geometry_ELP(count_ELP,15);        % inner height of single E core, m
    Ae_EE = EE_width_mid*EE_length;
    le_EE = (EE_width_outer+EE_width_inner)/2+2*(E_height_outer+E_height_inner);
    Ve_E = 2 * EE_length * (EE_width_outer * E_height_outer - (EE_width_inner...
                    - EE_width_mid) * E_height_inner);
    Al_EE = permeability_vacuum*rela_permeability_core*Ae_EE/le_EE;
% Calculate circuit states

    [I_ac_p_rms,I_sw_inner_rms_actual,I_sw_outer_rms_actual,...
        I_dio_clp_ave_actual,I_dio_addi_ave_actual,I_tr_p_rms_actual,...
        I_tr_s_rms_actual,I_dio_rec_ave_ideal,I_Lf_rms,V_Lac_PS,...
        V_Lac_delta_t1,V_Lac_delta_t2,V_trp_PS,V_trp_delta_t1,...
        V_trp_delta_t2,V_Lf_PS_delta_t1,V_Lf_delta_t2,delta_t1,...
        delta_t2,I_3,I_2,D_loss,I_Lf_ripple,ratio_div_sw_actual] = circuit_state(V_in,V_o,I_o,L_f,N_p,...
                                    N_s,L_ac_1,L_ac_2,f_s,PS);
                                
% transformer macro parameters
    R_T_p = 2 * Nstack_Tr * EE_length / ((I_tr_p_rms_actual / J) ...
            * sigma_copper);  % transformer primary winding resistance per turn, ohm
    R_T_s = 2 * Nstack_Tr * EE_length / ((I_tr_s_rms_actual / J) ...
            * sigma_copper);  % transformer secondary winding resistance per turn, ohm
    R_lkp = R_T_p * N_p; % transformer primary winding resistance, ohm
    R_lks = R_T_s * N_s; % transformer secondary winding resistance, ohm
    dia_p = 2 * sqrt((I_tr_p_rms_actual / J) / pi); % transformer pri winding diameter, m
    dia_s = 2 * sqrt((I_tr_s_rms_actual / J) / pi); % transformer sec winding diameter, m
    len_lk = EE_width_inner;    % length of leakage wire
    L_kp_T = 200e-9 * len_lk * (log((2 * len_lk / dia_p) * (1 + sqrt(1 + ...
        (dia_p / (2 * len_lk)) ^ 2))) - sqrt(1 + (dia_p / (2 * len_lk)) ...
        ^ 2) + 1 / 4 + dia_p / (2 * len_lk)); % pri leakage per turn
    L_ks_T = 200e-9 * len_lk * (log((2 * len_lk / dia_s) * (1 + sqrt(1 + ...
        (dia_s / (2 * len_lk)) ^ 2))) - sqrt(1 + (dia_s / (2 * len_lk)) ...
        ^ 2) + 1 / 4 + dia_s / (2 * len_lk)); % sec leakage per turn
    L_kp = L_kp_T * N_p;    % transformer primary leakage inductance, H
    L_ks = L_ks_T * N_s;    % transformer secondary leakage inductance, H
    C_p = (12 / N_p) ^ 2 * 16e-12;   % transformer primary parasitic capacitance, F
    C_s = (12 / N_s) ^ 2 * 10e-12;   % transformer secondary parasitic capacitance, F
    
% ac inductor macro parameters
    R_T_Lac = 2 * Nstack_Lac * EE_length / ((I_ac_p_rms / J) ...
            * sigma_copper);  % ac inductor winding resistance per turn, ohm
    Rm_core = le_EE / (permeability_vacuum * rela_permeability_core ...
                * Nstack_Lac * Ae_EE);
    I_peak_ac = 2 * I_3;
    B_m = 100e-3;
    N_T_Lac = ceil((L_ac * I_peak_ac) / (B_m * Nstack_Lac * Ae_EE));
    l_g = permeability_vacuum * Nstack_Lac * Ae_EE * (N_T_Lac ^ 2 / ...
                        L_ac - Rm_core);
    N_T_Lac_1 = N_T_Lac * sqrt(L_ac_ratio / (L_ac_ratio + 1));  % number of turns of external inductor 1
    N_T_Lac_2 = N_T_Lac * sqrt(1 / (L_ac_ratio + 1));  % number of turns of external inductor 2
    R_Lac_1 = R_T_Lac * N_T_Lac_1;    % ohm
    R_Lac_2 = R_T_Lac * N_T_Lac_2;    % ohm 
    
% dc inducor macro parameters
    Nstack_Ldc = 2; % number of stacked toroid cores
    toroid_OD = 33.66e-3;  % outer diamater, m
    toroid_ID = 19.4e-3;    % inner diameter, m
    toroid_HT = 11.5e-3;    % height, m
    toroid_le = 81.4e-3;    % path length, m
    toroid_Ae = 65.6e-6;    %m2
    toroid_Ve = 5340e-9;    % m3
    toroid_u = powder_u_dc_bias(Ldc_material,1);
    R_T_Lf = (Nstack_Ldc * 2 * toroid_HT + toroid_OD - toroid_ID) ...
        / ((I_Lf_rms / J) * sigma_copper);  % dc inductor winding resistance per turn, ohm
    Ldc_AL = permeability_vacuum * toroid_u * Nstack_Ldc * toroid_Ae / toroid_le; % AL value of powder core
    N_T_Lf = ceil(sqrt(L_f/Ldc_AL));      % number of turns of filter inductor
    Ldc_bias = (N_T_Lf * I_o / toroid_le) / 79.58;  % in oersted
    roll_off_bias = 1 / (powder_u_dc_bias(Ldc_material,2) + powder_u_dc_bias(Ldc_material,3) ...
        * Ldc_bias ^ powder_u_dc_bias(Ldc_material,4)); % percentage of initial permeability
    percent_fs = powder_u_fs(Ldc_material,2) + powder_u_fs(Ldc_material,3) * (f_s * 1e-6) ...
        + powder_u_fs(Ldc_material,4) * (f_s * 1e-6) ^ 2 + powder_u_fs(Ldc_material,5) * (f_s * 1e-6) ^ 3 ...
        + powder_u_fs(Ldc_material,6) * (f_s * 1e-6) ^ 4;
    percent_Ta = powder_u_Ta(Ldc_material,2) + powder_u_Ta(Ldc_material,3) * Ta ...
        + powder_u_Ta(Ldc_material,4) * Ta ^ 2 + powder_u_Ta(Ldc_material,5) * Ta ^ 3 ...
        + powder_u_Ta(Ldc_material,6) * Ta ^ 4;
    L_f_updated = L_f * (roll_off_bias / 100) * (1 + percent_fs) * (1 + percent_Ta);
    delta_Lf_percent = (L_f_updated - L_f) / L_f;
    
% Switching performance
% Rise time of switching, inner and outer switches

    [RT_inner,RT_outer] = Switching_performance(C_oss,ratio_div_sw_actual,I_3,V_in);
%     performance_data(p,number_of_param + 1) = RT_inner;
%     performance_data(p,number_of_param + 2) = RT_outer;
% Voltage gain
% calculate output-to-input voltage ratio

    [V_gain] = Vgain(N_p,N_s,PS,D_loss);
%     performance_data(p,number_of_param + 3) = V_gain;
% Semiconductor loss

    [P_cond_sw_inner,P_cond_sw_outer,P_cond_dio_clp,P_cond_dio_addi,...
        P_cond_dio_rec] = Semiconductor_loss(I_sw_inner_rms_actual,...
        I_sw_outer_rms_actual,I_dio_clp_ave_actual,I_dio_addi_ave_actual,...
        I_dio_rec_ave_ideal,R_on,V_d_pri,V_d_rec);
%     performance_data(p,number_of_param + 4) = P_cond_sw_inner;
%     performance_data(p,number_of_param + 5) = P_cond_sw_outer;
%     performance_data(p,number_of_param + 6) = P_cond_dio_clp;
%     performance_data(p,number_of_param + 7) = P_cond_dio_addi;
%     performance_data(p,number_of_param + 8) = P_cond_dio_rec;
% Primary oscillations

    [Osc_pri_peak] = Osc_tr_pri(C_p,L_kp,L_ks,L_ac_1,L_ac_2,N_p,N_s,...
                        V_in,RT_inner,f_s,R_Lac_1,R_Lac_2);
%     performance_data(p,number_of_param + 9) = Osc_pri_peak;
% Secondary oscillations

    [P_snubber,R_snubber] = Osc_tr_sec(C_p,C_s,L_kp,L_ks,...
                                        L_ac_1,L_ac_2,N_p,N_s,V_in,...
                                        R_Lac_1,R_Lac_2,R_lkp,...
                                        R_lks,Lac_s,C_r,I_2,f_s,...
                                        V_o,V_cp);
%     performance_data(p,number_of_param + 10) = P_snubber;
%     performance_data(p,number_of_param + 11) = R_snubber;
% Transformer loss

    % estimate Steinmetz coefficients
    f_st = ferrite_material(count_ferrite,2:4);  % frequency, kHz
    B_st = ferrite_material(count_ferrite,5:7);    % flux density, T
    Pv_st = ferrite_material(count_ferrite,8:10);   % core loss volume density, kW/m3
    [core_k,core_a,core_b] = SteinmetzConst(f_st,B_st,Pv_st);        
    
    Ae = Nstack_Tr * Ae_EE; % truncational area of core, m2
    Ve = Nstack_Tr * Ve_E; % volume of core, m3
    
    [P_winding_Trp,P_winding_Trs,P_core_Tr] = transformer_loss(...
                    I_tr_p_rms_actual,I_tr_s_rms_actual,V_trp_PS,...
                    V_trp_delta_t1,V_trp_delta_t2,R_T_p,R_T_s,f_s,...
                    PS,delta_t1,delta_t2,core_k,core_a,core_b,N_p,N_s,Ae,Ve);
%     performance_data(p,number_of_param + 12) = P_winding_Trp;
%     performance_data(p,number_of_param + 13) = P_winding_Trs;
%     performance_data(p,number_of_param + 14) = P_core_Tr;
% AC inductor loss

    % estimate Steinmetz coefficients
    f_st = ferrite_material(count_ferrite,2:4);  % frequency, kHz
    B_st = ferrite_material(count_ferrite,5:7);    % flux density, T
    Pv_st = ferrite_material(count_ferrite,8:10);   % core loss volume density, kW/m3
    [core_k,core_a,core_b] = SteinmetzConst(f_st,B_st,Pv_st);        
    
    Ae = Nstack_Lac * Ae_EE; % truncational area of core, m2
    Ve = Nstack_Lac * Ve_E; % volume of core, m3
    
    [P_winding_Lac,P_core_Lac] = ac_inductor_loss(I_ac_p_rms,V_Lac_PS,...
                                    V_Lac_delta_t1,V_Lac_delta_t2,R_T_Lac,f_s,...
                                    PS,delta_t1,delta_t2,core_k,core_a,core_b,...
                                    N_T_Lac,Ae,Ve);
%     performance_data(p,number_of_param + 15) = P_winding_Lac;
%     performance_data(p,number_of_param + 16) = P_core_Lac;
% DC inductor loss

    core_k = powder_core_loss(Ldc_material,2);    % kW/m3
    core_a = powder_core_loss(Ldc_material,4);
    core_b = powder_core_loss(Ldc_material,3);
    
    Ae = Nstack_Ldc * toroid_Ae; % truncational area of core, m2
    Ve = Nstack_Ldc * toroid_Ve; % volume of core, m3
                                
    [P_winding_Ldc,P_core_Ldc] = dc_inductor_loss(I_Lf_rms,...
                            V_Lf_PS_delta_t1,V_Lf_delta_t2,R_T_Lf,f_s,PS,...
                            delta_t1,delta_t2,core_k,core_a,core_b,N_T_Lf,...
                            Ae,Ve);
%     performance_data(p,number_of_param + 17) = P_winding_Ldc;
%     performance_data(p,number_of_param + 18) = P_core_Ldc;
    
% Other performance index

%     performance_data(p,number_of_param + 19) = D_loss;
%     performance_data(p,number_of_param + 20) = I_Lf_ripple;
%     performance_data(p,number_of_param + 21) = delta_Lf_percent;
% End of loop

    performance_data(p,:) = [Para_ana(p,:) RT_inner RT_outer V_gain ...
        P_cond_sw_inner P_cond_sw_outer P_cond_dio_clp P_cond_dio_addi ...
        P_cond_dio_rec Osc_pri_peak P_snubber R_snubber P_winding_Trp ...
        P_winding_Trs P_core_Tr P_winding_Lac P_core_Lac P_winding_Ldc ...
        P_core_Ldc D_loss I_Lf_ripple delta_Lf_percent I_ac_p_rms];

end

spmd
    cd(currDir);
    rmdir(tmpDir,'s');
    rmpath(currDir);
    close_system('Osc_sec', 0);
end

close_system('Osc_sec', 0);
delete(gcp('nocreate'));

elapsedTime = toc;
disp(['elapsedTime is ',num2str(elapsedTime),' seconds.']);
%save('brute_force_search_Ta.mat','performance_data','number_of_perf','number_of_param');
%%
% Performance score
%

Passing_score_V_gain = 0.52;
Passing_score_P_cond_sw_inner = 7;
Passing_score_P_cond_dio_addi = 1.2;
Passing_score_P_cond_dio_rec = 20;
Passing_score_Osc_pri_peak = 50;
Passing_score_P_snubber = 21;
Passing_score_P_core_Tr = 10;
Passing_score_P_core_Lac = 5;

%   Check scores
Score_index = performance_data(:,number_of_param + 3) < Passing_score_V_gain;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 4) > Passing_score_P_cond_sw_inner;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 7) > Passing_score_P_cond_dio_addi;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 8) > Passing_score_P_cond_dio_rec;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 9) > Passing_score_Osc_pri_peak;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 10) > Passing_score_P_snubber;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 14) > Passing_score_P_core_Tr;
performance_data(Score_index,:) = [];

Score_index = performance_data(:,number_of_param + 16) > Passing_score_P_core_Lac * 2;
performance_data(Score_index,:) = [];

%% FEMM simulations
% parameters to sweep
Para_circuit = 1:1:size(performance_data,1);
Para_N_B = 4:1:4;    % number of PCBs
Para_winding_P_width = 2.1:0.2:2.5; % mm
Para_Winding_P_dist = 0.6:0.2:1; % mm
Para_winding_S_width = 1.6:0.2:2; % mm
Para_Winding_S_dist = 0.5:0.1:0.7; % mm

Result_FEMM = zeros(size(Para_circuit,2) * ...
    size(Para_N_B,2) * ...
    size(Para_winding_P_width,2) * ...
    size(Para_Winding_P_dist,2) * ...
    size(Para_winding_S_width,2) * ...
    size(Para_Winding_S_dist,2),9);

para_FEMM = zeros(size(Para_circuit,2) * ...
    size(Para_N_B,2) * ...
    size(Para_winding_P_width,2) * ...
    size(Para_Winding_P_dist,2) * ...
    size(Para_winding_S_width,2) * ...
    size(Para_Winding_S_dist,2),6);

count = 0;
for Macrocircuit = Para_circuit
    for N_B = Para_N_B
        for winding_P_width = Para_winding_P_width
            for Winding_P_dist = Para_Winding_P_dist
                for winding_S_width = Para_winding_S_width
                    for Winding_S_dist = Para_Winding_S_dist

                            Para = [Macrocircuit N_B winding_P_width ...
                                Winding_P_dist winding_S_width ...
                                Winding_S_dist];

                            count = count + 1;

                            para_FEMM(count,:) = Para;
                    end
                end
            end
        end
    end
end

% fixed parameters
    winding_th_FEMM = 0.07; % mm
    D_layer_layer_FEMM = 0.7; % mm, thickness of PCB, layer to layer distance
    Num_worker_FEMM = 20; % number of worker for parallel computing FEMM
    
    % free space margin
    margin = 5; % mm
    
    % bump geometric parameters
%    bump_length = 0; % mm
    bump_th = 0.2; % mm, thickness of bump
    bump_length = zeros(1,8);

%set up parallel pool
 
    parpool('local',Num_worker_FEMM);

%    global HandleToFEMM
    
parfor p = 1:count
    
    Pid = getCurrentTask(); % get id of current worker
    
    ELP_No_FEMM = performance_data(para_FEMM(p,1),7);
    N_stack_FEMM = performance_data(para_FEMM(p,1),8);
    N_P_L_FEMM = performance_data(para_FEMM(p,1),9) / para_FEMM(p,2);  % number of primary turns per layer
    N_S_L_FEMM = N_P_L_FEMM / performance_data(para_FEMM(p,1),4);  % number of secondary turns per layer
    I_P_FEMM = performance_data(para_FEMM(p,1),35); % A, primary RMS
    f_s_FEMM = performance_data(para_FEMM(p,1),1);
    
    N_B = para_FEMM(p,2);
    winding_P_width = para_FEMM(p,3);
    Winding_P_dist = para_FEMM(p,4);
    winding_S_width = para_FEMM(p,5);
    Winding_S_dist = para_FEMM(p,6);
    
    E_length_O_FEMM = geometry_ELP(ELP_No_FEMM,10) * 1000; % mm
    E_length_I_FEMM = geometry_ELP(ELP_No_FEMM,14) * 1000; % mm
    E_width_FEMM = geometry_ELP(ELP_No_FEMM,12) * 1000; % mm
    E_mid_FEMM = geometry_ELP(ELP_No_FEMM,13) * 1000; % mm
    E_height_O_FEMM = geometry_ELP(ELP_No_FEMM,11) * 1000; % mm
    E_height_I_FEMM = geometry_ELP(ELP_No_FEMM,15) * 1000; % mm
    
    Window_width = (E_length_I_FEMM - E_mid_FEMM) / 2; % mm, width of window
    Window_height = E_height_I_FEMM * 2; % mm, height of window    
    
    % geometric parameters of winding    
    
    D_Horiz_P = (Window_width - N_P_L_FEMM * winding_P_width...
        - (N_P_L_FEMM - 1) * Winding_P_dist) / 2; % primary winding to core 
        % distance
    D_Horiz_S = (Window_width - N_S_L_FEMM * winding_S_width...
        - (N_S_L_FEMM - 1) * Winding_S_dist) / 2; % secondary winding to core 
        % distance
  
    % vertical board surface to surce distance
    D_vertical = (Window_height - D_layer_layer_FEMM * N_B) / (N_B + 1); 
    
    % excitation
    I_S = I_P_FEMM * N_P_L_FEMM / N_S_L_FEMM; % A, secondary RMS

    HandleToFEMM = Pid.ID;
% open femm with main window hiden
    openfemm(1);    
    %hand = HandleToFEMM;
% create a new preprocessor document: magnetics problem

    newdocument(0);

% define the problem
    % mi_probdef(freq,units,type,precision,depth,minangle,(acsolver))

    % freq: in Hertz

    % units: e 'inches', 'millimeters', 'centimeters', 'mils', 'meters', 
    % and 'micrometers'

    % type:  'planar' - 2-D planar problem, 'axi' - axisymmetric problem

    % precision: maixmum of RMS of the residual

    % depth: the depth of the problem in the into-the-page direction for
    % 2-D planar problems

    % minable:  the minimum angle constraint sent to the mesh generator

    % (acsolver): solver is to be used for AC problems: 0 for successive 
    % approximation, 1 for Newton.

    mi_probdef(f_s_FEMM,'millimeters','planar',1e-8,E_width_FEMM*N_stack_FEMM,30,0);

% create geometry

    % core
    
    % outer profile
    mi_drawrectangle([0 0; E_length_O_FEMM / 2 2 * E_height_O_FEMM]);
    
    % inner profile (with FCT)
    mi_drawpolygon([E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM; ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2; ...
        E_mid_FEMM / 2 + bump_length(1),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2; ...
        E_mid_FEMM / 2 + bump_length(1),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2; ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2; ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(2),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(2),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(3),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(3),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(4),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2 + 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + bump_length(4),E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM / 2 + bump_th / 2 + 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2,E_height_O_FEMM + E_height_I_FEMM; ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM; ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2); ...
        E_mid_FEMM / 2 + Window_width - bump_length(5),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2); ...
        E_mid_FEMM / 2 + Window_width - bump_length(5),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th; ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th; ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(6),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(6),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 1 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(7),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(7),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 2 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(8),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width - bump_length(8),E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM + E_height_I_FEMM - (D_vertical + D_layer_layer_FEMM / 2 - bump_th / 2) - bump_th - 3 * (D_layer_layer_FEMM + D_vertical); ...
        E_mid_FEMM / 2 + Window_width,E_height_O_FEMM - E_height_I_FEMM]);
    
    % winding
    for j = 1:1:N_B
        
        % primary winding
        if mod(j,2)
            for i = 1:1:N_P_L_FEMM 

                mi_drawrectangle([E_mid_FEMM / 2 + D_Horiz_P + (i - 1) * (winding_P_width + Winding_P_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM + (j - 1) * (D_layer_layer_FEMM + D_vertical); ...
                    E_mid_FEMM / 2 + D_Horiz_P + winding_P_width + (i - 1) * (winding_P_width + Winding_P_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + (j - 1) * (D_layer_layer_FEMM + D_vertical)]);

                mi_addblocklabel(E_mid_FEMM / 2 + D_Horiz_P + winding_P_width / 2 + (i - 1) * (winding_P_width + Winding_P_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical));

                mi_addcircprop(['coil_B_' num2str(j) '_P_' num2str(i)], I_P_FEMM, 1);

                mi_selectlabel(E_mid_FEMM / 2 + D_Horiz_P + winding_P_width / 2 + (i - 1) * (winding_P_width + Winding_P_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical));
                mi_setblockprop('Copper', 0, 1, ['coil_B_' num2str(j) '_P_' num2str(i)], 0, 0, 0);
                mi_clearselected;

            end
        else
            for i = 1:1:N_P_L_FEMM 

                mi_drawrectangle([E_mid_FEMM / 2 + D_Horiz_P + (i - 1) * (winding_P_width + Winding_P_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM + (j - 1) * (D_layer_layer_FEMM + D_vertical) + D_layer_layer_FEMM + winding_th_FEMM; ...
                    E_mid_FEMM / 2 + D_Horiz_P + winding_P_width + (i - 1) * (winding_P_width + Winding_P_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + (j - 1) * (D_layer_layer_FEMM + D_vertical) + D_layer_layer_FEMM + winding_th_FEMM]);

                mi_addblocklabel(E_mid_FEMM / 2 + D_Horiz_P + winding_P_width / 2 + (i - 1) * (winding_P_width + Winding_P_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical) + D_layer_layer_FEMM + winding_th_FEMM);

                mi_addcircprop(['coil_B_' num2str(j) '_P_' num2str(i)], I_P_FEMM, 1);

                mi_selectlabel(E_mid_FEMM / 2 + D_Horiz_P + winding_P_width / 2 + (i - 1) * (winding_P_width + Winding_P_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical - winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical) + D_layer_layer_FEMM + winding_th_FEMM);
                mi_setblockprop('Copper', 0, 1, ['coil_B_' num2str(j) '_P_' num2str(i)], 0, 0, 0);
                mi_clearselected;

            end
        end

        % secondary winding
        if mod(j,2)
            for i = 1:1:N_S_L_FEMM 

                mi_drawrectangle([E_mid_FEMM / 2 + D_Horiz_S + (i - 1) * (winding_S_width + Winding_S_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM + (j - 1) * (D_layer_layer_FEMM + D_vertical); ...
                    E_mid_FEMM / 2 + D_Horiz_S + winding_S_width + (i - 1) * (winding_S_width + Winding_S_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM + winding_th_FEMM + (j - 1) * (D_layer_layer_FEMM + D_vertical)]);

                mi_addblocklabel(E_mid_FEMM / 2 + D_Horiz_S + winding_S_width / 2 + (i - 1) * (winding_S_width + Winding_S_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM + winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical));

                mi_addcircprop(['coil_B_' num2str(j) '_S_' num2str(i)], - I_S, 1);

                mi_selectlabel(E_mid_FEMM / 2 + D_Horiz_S + winding_S_width / 2 + (i - 1) * (winding_S_width + Winding_S_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + D_vertical + D_layer_layer_FEMM + winding_th_FEMM / 2 + (j - 1) * (D_layer_layer_FEMM + D_vertical));
                mi_setblockprop('Copper', 0, 1, ['coil_B_' num2str(j) '_S_' num2str(i)], 0, 0, 0);
                mi_clearselected;

            end
        else
            for i = 1:1:N_S_L_FEMM 

                mi_drawrectangle([E_mid_FEMM / 2 + D_Horiz_S + (i - 1) * (winding_S_width + Winding_S_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + j * (D_layer_layer_FEMM + D_vertical) - D_layer_layer_FEMM - winding_th_FEMM; ...
                    E_mid_FEMM / 2 + D_Horiz_S + winding_S_width + (i - 1) * (winding_S_width + Winding_S_dist) ...
                    E_height_O_FEMM - E_height_I_FEMM + winding_th_FEMM + j * (D_layer_layer_FEMM + D_vertical) - D_layer_layer_FEMM - winding_th_FEMM]);

                mi_addblocklabel(E_mid_FEMM / 2 + D_Horiz_S + winding_S_width / 2 + (i - 1) * (winding_S_width + Winding_S_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + winding_th_FEMM / 2 + j * (D_layer_layer_FEMM + D_vertical) - D_layer_layer_FEMM - winding_th_FEMM);

                mi_addcircprop(['coil_B_' num2str(j) '_S_' num2str(i)], - I_S, 1);

                mi_selectlabel(E_mid_FEMM / 2 + D_Horiz_S + winding_S_width / 2 + (i - 1) * (winding_S_width + Winding_S_dist),...
                    E_height_O_FEMM - E_height_I_FEMM + winding_th_FEMM / 2 + j * (D_layer_layer_FEMM + D_vertical) - D_layer_layer_FEMM - winding_th_FEMM);
                mi_setblockprop('Copper', 0, 1, ['coil_B_' num2str(j) '_S_' num2str(i)], 0, 0, 0);
                mi_clearselected;

            end
        end
        
    
    end
    
    % air space, as boundary
    mi_drawrectangle([-margin -margin; E_length_O_FEMM / 2 + margin ...
        2 * E_height_O_FEMM + margin]);
    
% Add block labels

    % core
    mi_addblocklabel(E_mid_FEMM / 4,(E_height_O_FEMM - E_height_I_FEMM) / 2);
    
    % air window
    mi_addblocklabel(E_length_O_FEMM / 4,E_height_O_FEMM - E_height_I_FEMM + (D_vertical - winding_th_FEMM) / 2);
    
    % air space, as boundary
    mi_addblocklabel(- margin / 2,- margin / 2);
    
    % block labels for coil have been added when creating geometry
    
% Define boundary condition

    mi_addboundprop('BC', 0, 0, 0, 0, 0, 0, 0, 0, 0);
    
% Apply boundary condition

    mi_selectsegment(- margin,E_height_O_FEMM); 
    mi_selectsegment(E_length_O_FEMM / 2 + margin,E_height_O_FEMM); 
    mi_selectsegment(E_length_O_FEMM / 4,- margin);
    mi_selectsegment(E_length_O_FEMM / 4,2 * E_height_O_FEMM + margin);
    mi_setsegmentprop('BC', 1, 1, 0, 0);
    mi_clearselected;
    
% Add material

    mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
    mi_addmaterial('Iron', 2600, 2600, 0, 0, 0, 0, 0, 1, 0, 0, 0);
    mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0);
    
% Add circuit property, have done when creating geometry
    
% Apply the materials to the appropriate block label

    mi_selectlabel(- margin / 2,- margin / 2);
    mi_setblockprop('Air', 0, 1, '<None>', 0, 0, 0);
    mi_clearselected;
    
    mi_selectlabel(E_length_O_FEMM / 4,E_height_O_FEMM - E_height_I_FEMM + (D_vertical - winding_th_FEMM) / 2);
    mi_setblockprop('Air', 0, 1, '<None>', 0, 0, 0);
    mi_clearselected;
    
    mi_selectlabel(E_mid_FEMM / 4,(E_height_O_FEMM - E_height_I_FEMM) / 2);
    mi_setblockprop('Iron', 0, 1, '<None>', 0, 0, 0);
    mi_clearselected;
    
% save file
    mi_saveas(['Tr_FCT_' num2str(Pid.ID) '.fem']);
    
% analyze and load solution

    mi_analyze;
    mi_loadsolution;
    
%     
% get current, voltage, and flux linkage of coils

    primary_coils = zeros(N_P_L_FEMM * N_B,4);
    secondary_coils = zeros(N_S_L_FEMM * N_B,4);

    for j = 1:1:N_B
        
        % primary winding
        for i = 1:1:N_P_L_FEMM 

            vals = mo_getcircuitproperties(['coil_B_' num2str(j) '_P_' num2str(i)]);
            primary_coils(i + N_P_L_FEMM * (j - 1),1) = vals(1);
            primary_coils(i + N_P_L_FEMM * (j - 1),2) = vals(2);
            primary_coils(i + N_P_L_FEMM * (j - 1),3) = vals(3);
            primary_coils(i + N_P_L_FEMM * (j - 1),4) = real(vals(2) / vals(1));

        end

        % secondary winding
        for i = 1:1:N_S_L_FEMM 

            vals = mo_getcircuitproperties(['coil_B_' num2str(j) '_S_' num2str(i)]);
            secondary_coils(i + N_S_L_FEMM * (j - 1),1) = vals(1);
            secondary_coils(i + N_S_L_FEMM * (j - 1),2) = vals(2);
            secondary_coils(i + N_S_L_FEMM * (j - 1),3) = vals(3);
            secondary_coils(i + N_S_L_FEMM * (j - 1),4) = real(vals(2) / vals(1));

        end
    
    end
        
    R_pri = sum(primary_coils(:,4));
    R_sec = sum(secondary_coils(:,4));
    Loss = I_P_FEMM * I_P_FEMM * R_pri + I_S * I_S * R_sec;
%     
    closefemm;
    
    Result_FEMM(p,:) = [para_FEMM(p,:) R_pri R_sec Loss];
    
end
delete(gcp('nocreate'));
%% FEMM post processing
[Min,min_index] = min(Result_FEMM(:,9));
Winding_Pri_resistance_FEMM = Result_FEMM(min_index,7);
Winding_Sec_resistance_FEMM = Result_FEMM(min_index,8);
Winding_loss_FEMM = Result_FEMM(min_index,9);
% parameter set for 3D comsol simulation
Para_COMSOL = [performance_data(Result_FEMM(min_index,1),[1:13 35]) Result_FEMM(min_index,2:6)];
%% COMOSL simulation
        f_s_COMSOL = Para_COMSOL(1);
        Turns_ratio_COMSOL = Para_COMSOL(4);  % primary to secondary turns ratio
        I_rms_P_COMSOL = Para_COMSOL(14);        % primary rms current, A

        EE_width_outer = geometry_ELP(Para_COMSOL(7),10);       % outer width of the core studied, m
        EE_width_inner = geometry_ELP(Para_COMSOL(7),14);       % inner width of the core studied, m
        EE_width_mid = geometry_ELP(Para_COMSOL(7),13);            % width of mid leg of the core  studied, m
        EE_length = geometry_ELP(Para_COMSOL(7),12) * Para_COMSOL(8);
        %EE_length = geometry_ELP(Para_COMSOL(7),12);
        E_height_outer = geometry_ELP(Para_COMSOL(7),11);        % outer height of single E core, m
        E_height_inner = geometry_ELP(Para_COMSOL(7),15);        % inner height of single E core, m
                
        N_P = Para_COMSOL(9);
        N_S = N_P / Turns_ratio_COMSOL;            % number of turns of secondary winding
        
        N_B = Para_COMSOL(15);
        N_P_L = N_P / N_B;    % number of primary turns on each layer
        N_S_L = N_S / N_B;
        %
        width_Pri = Para_COMSOL(16) / 1000;
        width_Sec = Para_COMSOL(18) / 1000;
        
        D_cond_to_cond_P = Para_COMSOL(17) / 1000;      % distance between primary conductors on the same layer, m
        D_cond_to_cond_S = Para_COMSOL(19) / 1000;      % distance between secondary conductors on the same layer, m
        thickness_board = D_layer_layer_FEMM / 1000;      % thickness of board, m
        thickness_cond = winding_th_FEMM / 1000;    % thickness of conductor, m
        air_factor = 2;
        %
        D_cond_to_core_P = ((EE_width_inner-EE_width_mid)/2-N_P_L*width_Pri-(N_P_L-1)*D_cond_to_cond_P)/2;      % perpendicular distance between conductor and core, m
        D_cond_to_core_S = ((EE_width_inner-EE_width_mid)/2-N_S_L*width_Sec-(N_S_L-1)*D_cond_to_cond_S)/2;
        D_cond_to_core = (D_cond_to_core_P + D_cond_to_core_S) / 2;
        %
        % element properties
        maximum_size_winding = 10*thickness_cond;  %m
        minimum_size_winding = thickness_cond;    %m
        max_growth_rate_winding = 1.4;
        curvature_factor_winding = 0.4;
        resol_narrow_widning = 0.7;
        number_e_face_widning = 2; % number of distributed elements
        %
        maximum_size_general = 20*D_cond_to_core;  %m
        minimum_size_general = 5*D_cond_to_core;    %m
        max_growth_rate_general = 1.4;
        curvature_factor_general = 0.4;
        resol_narrow_general = 0.7;
        %
        rela_permeability_core = 1500;
        % intermediate parameter
        I_rms_S = I_rms_P_COMSOL*(N_P_L/N_S_L);  % secondary rms current, A       
        N_B_total = N_S/N_S_L;      % total number of two-layer PCBs
        distance_B_to_B = (2*E_height_inner-N_B_total*thickness_board)/(N_B_total+1);
        %%
        if distance_B_to_B > 10*thickness_cond
            displacement_B_to_B = distance_B_to_B+thickness_board;
            %%
            import com.comsol.model.*
            import com.comsol.model.util.* % import the COMSOL class
            ModelUtil.clear;
            %ModelUtil.showProgress(true);       % enable a progress bar and visualize the progress of operations
            ModelUtil.create('transformer_ELP');  % Use the method ModelUtil.create to create a new model object 'transformer_ELP' on the COMSOL server
            model_transformer_ELP = ModelUtil.model('transformer_ELP'); % create a MATLAB variable 'transformer_ELP' that is linked to the 'transformer_ELP' model object.
            model_transformer_ELP.component.create('comp1');    % creat a new component node 'comp1' in model object 'transformer_ELP'
            comp_transformer_ELP = model_transformer_ELP.component('comp1'); % create a MATLAB variable 'comp_transformer_ELP' that is linked to the 'comp1' component node. 
            comp_transformer_ELP.geom.create('geom1', 3);   % add a new 3D geometry 'geom1' to the component node 'comp_transformer_ELP'
    %%
    % create materials
            %
            comp_transformer_ELP.material.create('mat1', 'Common');
            comp_transformer_ELP.material('mat1').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');
            comp_transformer_ELP.material('mat1').propertyGroup.create('linzRes', 'Linearized resistivity');
            comp_transformer_ELP.material('mat1').label('Copper');
            comp_transformer_ELP.material('mat1').set('family', 'copper');
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('electricconductivity', {'5.998e7[S/m]' '0' '0' '0' '5.998e7[S/m]' '0' '0' '0' '5.998e7[S/m]'});
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('thermalexpansioncoefficient', {'17e-6[1/K]' '0' '0' '0' '17e-6[1/K]' '0' '0' '0' '17e-6[1/K]'});
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('heatcapacity', '385[J/(kg*K)]');
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('density', '8960[kg/m^3]');
            comp_transformer_ELP.material('mat1').propertyGroup('def').set('thermalconductivity', {'400[W/(m*K)]' '0' '0' '0' '400[W/(m*K)]' '0' '0' '0' '400[W/(m*K)]'});
            comp_transformer_ELP.material('mat1').propertyGroup('Enu').set('youngsmodulus', '110e9[Pa]');
            comp_transformer_ELP.material('mat1').propertyGroup('Enu').set('poissonsratio', '0.35');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('rho0', '');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('alpha', '');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('Tref', '');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('rho0', '1.72e-8[ohm*m]');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('alpha', '0.0039[1/K]');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').set('Tref', '298[K]');
            comp_transformer_ELP.material('mat1').propertyGroup('linzRes').addInput('temperature');
            comp_transformer_ELP.material('mat1').set('family', 'copper');
            %
            comp_transformer_ELP.material.create('mat2', 'Common');
            comp_transformer_ELP.material('mat2').label('Soft Iron (With Losses)');
            comp_transformer_ELP.material('mat2').set('family', 'iron');
            comp_transformer_ELP.material('mat2').propertyGroup('def').set('electricconductivity', {'0.1[S/m]' '0' '0' '0' '0.1[S/m]' '0' '0' '0' '0.1[S/m]'});
            comp_transformer_ELP.material('mat2').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat2').propertyGroup('def').set('relpermeability', {num2str(rela_permeability_core)});
            %
            comp_transformer_ELP.material.create('mat3', 'Common');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('eta', 'Piecewise');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('Cp', 'Piecewise');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('rho', 'Analytic');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('k', 'Piecewise');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('cs', 'Analytic');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('an1', 'Analytic');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func.create('an2', 'Analytic');
            comp_transformer_ELP.material('mat3').propertyGroup.create('RefractiveIndex', 'Refractive index');
            comp_transformer_ELP.material('mat3').propertyGroup.create('NonlinearModel', 'Nonlinear model');
            comp_transformer_ELP.material('mat3').label('Air');
            comp_transformer_ELP.material('mat3').set('family', 'air');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('eta').set('arg', 'T');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('eta').set('pieces', {'200.0' '1600.0' '-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('eta').set('argunit', 'K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('eta').set('fununit', 'Pa*s');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('Cp').set('arg', 'T');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('Cp').set('pieces', {'200.0' '1600.0' '1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('Cp').set('argunit', 'K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('Cp').set('fununit', 'J/(kg*K)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('expr', 'pA*0.02897/R_const[K*mol/J]/T');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('args', {'pA' 'T'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('dermethod', 'manual');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('argders', {'pA' 'd(pA*0.02897/R_const/T,pA)'; 'T' 'd(pA*0.02897/R_const/T,T)'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('argunit', 'Pa,K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('fununit', 'kg/m^3');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('rho').set('plotargs', {'pA' '0' '1'; 'T' '0' '1'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('k').set('arg', 'T');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('k').set('pieces', {'200.0' '1600.0' '-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('k').set('argunit', 'K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('k').set('fununit', 'W/(m*K)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('expr', 'sqrt(1.4*R_const[K*mol/J]/0.02897*T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('args', {'T'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('dermethod', 'manual');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('argunit', 'K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('fununit', 'm/s');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('cs').set('plotargs', {'T' '273.15' '373.15'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('funcname', 'alpha_p');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('expr', '-1/rho(pA,T)*d(rho(pA,T),T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('args', {'pA' 'T'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('argunit', 'Pa,K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('fununit', '1/K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an1').set('plotargs', {'pA' '101325' '101325'; 'T' '273.15' '373.15'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('funcname', 'muB');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('expr', '0.6*eta(T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('args', {'T'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('argunit', 'K');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('fununit', 'Pa*s');
            comp_transformer_ELP.material('mat3').propertyGroup('def').func('an2').set('plotargs', {'T' '200' '1600'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('thermalexpansioncoefficient', '');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('molarmass', '');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('bulkviscosity', '');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('dynamicviscosity', 'eta(T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('ratioofspecificheat', '1.4');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('electricconductivity', {'1[S/m]' '0' '0' '0' '1[S/m]' '0' '0' '0' '1[S/m]'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('heatcapacity', 'Cp(T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('density', 'rho(pA,T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('thermalconductivity', {'k(T)' '0' '0' '0' 'k(T)' '0' '0' '0' 'k(T)'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('soundspeed', 'cs(T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('thermalexpansioncoefficient', {'alpha_p(pA,T)' '0' '0' '0' 'alpha_p(pA,T)' '0' '0' '0' 'alpha_p(pA,T)'});
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('molarmass', '0.02897');
            comp_transformer_ELP.material('mat3').propertyGroup('def').set('bulkviscosity', 'muB(T)');
            comp_transformer_ELP.material('mat3').propertyGroup('def').addInput('temperature');
            comp_transformer_ELP.material('mat3').propertyGroup('def').addInput('pressure');
            comp_transformer_ELP.material('mat3').propertyGroup('RefractiveIndex').set('n', '');
            comp_transformer_ELP.material('mat3').propertyGroup('RefractiveIndex').set('ki', '');
            comp_transformer_ELP.material('mat3').propertyGroup('RefractiveIndex').set('n', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
            comp_transformer_ELP.material('mat3').propertyGroup('RefractiveIndex').set('ki', {'0' '0' '0' '0' '0' '0' '0' '0' '0'});
            comp_transformer_ELP.material('mat3').propertyGroup('NonlinearModel').set('BA', '(def.gamma+1)/2');
            comp_transformer_ELP.material('mat3').materialType('nonSolid');
            comp_transformer_ELP.material('mat3').set('family', 'air');
    %%
    %
    % Create geometry and assign material
            % default units
            %               length - m
            geom_transformer_ELP = comp_transformer_ELP.geom('geom1'); % create a MATLAB variable 'geom_transformer_ELP' that is linked to the 'geom1' 3D geometry
            %
            % create geometry of core
            geom_transformer_ELP.create('wp1', 'WorkPlane');    % add a new feature - work plane 'wp1' to the geometry 'geom_transformer_ELP'
            wp1 = geom_transformer_ELP.feature('wp1');
            wp1.set('planetype', 'quick');
            wp1.set('quickplane', 'yz');
            wp1.set('quickx', 0);
            %
            wp1.geom.create('rec1', 'Rectangle');
            rec1 = wp1.geom.feature('rec1');
            rec1.set('pos', [-EE_width_outer/2 -E_height_outer]);
            rec1.set('size', [EE_width_outer 2*E_height_outer]);
            wp1.geom.run('rec1');
            %
            wp1.geom.create('rec2', 'Rectangle');
            rec2 = wp1.geom.feature('rec2');
            rec2.set('pos', [EE_width_mid/2 -E_height_inner]);
            rec2.set('size', [(EE_width_inner-EE_width_mid)/2 2*E_height_inner]);
            wp1.geom.run('rec2');
            %
            wp1.geom.create('rec3', 'Rectangle');
            rec3 = wp1.geom.feature('rec3');
            rec3.set('pos', [-EE_width_inner/2 -E_height_inner]);
            rec3.set('size', [(EE_width_inner-EE_width_mid)/2 2*E_height_inner]);
            wp1.geom.run('rec3');
            %
            wp1.geom.feature.create('core_face','Compose');   % add a new feature - compose 'core_face' 
            core_face = wp1.geom.feature('core_face');     % create a MATLAB variable 'core_face' that is linked to the compose 'core_face'
            core_face.selection('input').set({'rec1' 'rec2' 'rec3'});
            core_face.set('formula','rec1-rec2-rec3');
            wp1.geom.run('core_face');
            %
            geom_transformer_ELP.create('ext1', 'Extrude');     % add a new feature - extrude 'ext1'
            ext1 = geom_transformer_ELP.feature('ext1');        % create a MATLAB variable 'ext1' that is linked to the extrude 'ext1'
            ext1.selection('input').set({'wp1'});
            ext1.set('distance', EE_length);
            ext1.set('selresult', 'on');
            geom_transformer_ELP.run('ext1');                   % build extrude 'ext1'
            %
            comp_transformer_ELP.material('mat2').selection.named('geom1_ext1_dom');
            %
            % create geometry of winding
            Union_selection_copper = {};    % used to store all of the selections of conductor
            count_selection_copper = 0;
            %
            % create geometry of secondary winding
            for count_board = 1:1:N_B_total
                %
                wp_number = join(['B_' num2str(count_board) '_S']);
                geom_transformer_ELP.create(wp_number, 'WorkPlane');    % add a new feature - work plane 
                geom_transformer_ELP.feature(wp_number).set('planetype', 'quick');
                geom_transformer_ELP.feature(wp_number).set('quickplane', 'xy');
                geom_transformer_ELP.feature(wp_number).set('quickz', -E_height_inner+distance_B_to_B+(count_board-1)*displacement_B_to_B);
                %
                for count_S_L = 1:1:N_S_L
                    %
                    rec_number1 = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_rec_1']);
                    fil_number1 = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_fil_1']);
                    geom_transformer_ELP.feature(wp_number).geom.create(rec_number1, 'Rectangle');
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number1).set('pos', [-D_cond_to_core_S-(count_S_L-1)*(width_Sec+D_cond_to_cond_S) -(EE_width_mid/2+D_cond_to_core_S+(count_S_L-1)*(width_Sec+D_cond_to_cond_S))]);
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number1).set('size', [EE_length+2*D_cond_to_core_S+2*(count_S_L-1)*(width_Sec+D_cond_to_cond_S) EE_width_mid+2*D_cond_to_core_S+2*(count_S_L-1)*(width_Sec+D_cond_to_cond_S)]);
                    geom_transformer_ELP.feature(wp_number).geom.run(rec_number1);
                    geom_transformer_ELP.feature(wp_number).geom.create(fil_number1,'Fillet');
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number1).selection('point').set(join([rec_number1 '(1)']),1:4);
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number1).set('radius',D_cond_to_core_S);
                    geom_transformer_ELP.feature(wp_number).geom.run(fil_number1);
                    %
                    rec_number2 = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_rec_2']);
                    fil_number2 = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_fil_2']);
                    geom_transformer_ELP.feature(wp_number).geom.create(rec_number2, 'Rectangle');
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number2).set('pos', [-D_cond_to_core_S-width_Sec-(count_S_L-1)*(width_Sec+D_cond_to_cond_S) -(EE_width_mid/2+D_cond_to_core_S+width_Sec+(count_S_L-1)*(width_Sec+D_cond_to_cond_S))]);
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number2).set('size', [EE_length+2*D_cond_to_core_S+2*width_Sec+2*(count_S_L-1)*(width_Sec+D_cond_to_cond_S) EE_width_mid+2*D_cond_to_core_S+2*width_Sec+2*(count_S_L-1)*(width_Sec+D_cond_to_cond_S)]);
                    geom_transformer_ELP.feature(wp_number).geom.run(rec_number2);
                    geom_transformer_ELP.feature(wp_number).geom.create(fil_number2,'Fillet');
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number2).selection('point').set(join([rec_number2 '(1)']),1:4);
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number2).set('radius',D_cond_to_core_S);
                    geom_transformer_ELP.feature(wp_number).geom.run(fil_number2);
                    %
                    diff_winding = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_diff']);
                    geom_transformer_ELP.feature(wp_number).geom.create(diff_winding,'Difference');   
                    geom_transformer_ELP.feature(wp_number).geom.feature(diff_winding).selection('input').set({fil_number2});
                    geom_transformer_ELP.feature(wp_number).geom.feature(diff_winding).selection('input2').set({fil_number1});
                    geom_transformer_ELP.feature(wp_number).geom.run(diff_winding);
                    %
                end
                extrude_winding = join(['B_' num2str(count_board) '_S_ext']);
                geom_transformer_ELP.create(extrude_winding, 'Extrude');     % add a new feature - extrude 
                geom_transformer_ELP.feature(extrude_winding).selection('input').set({wp_number});
                geom_transformer_ELP.feature(extrude_winding).set('distance', -thickness_cond);
                geom_transformer_ELP.feature(extrude_winding).set('selresult', 'on');
                geom_transformer_ELP.run(extrude_winding);                   % build extrude
                count_selection_copper = count_selection_copper+1;
                %selection_name_copper = join([extrude_winding '_dom']);
                Union_selection_copper{count_selection_copper} = extrude_winding;
            end
            %
            % create geometry of primary winding
            for count_board = 1:1:N_B_total
                %
                wp_number = join(['B_' num2str(count_board) '_P']);
                geom_transformer_ELP.create(wp_number, 'WorkPlane');    % add a new feature - work plane 
                geom_transformer_ELP.feature(wp_number).set('planetype', 'quick');
                geom_transformer_ELP.feature(wp_number).set('quickplane', 'xy');
                geom_transformer_ELP.feature(wp_number).set('quickz', -E_height_inner+count_board*displacement_B_to_B);
                %
                for count_P_L = 1:1:N_P_L
                    %
                    rec_number1 = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_rec_1']);
                    fil_number1 = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_fil_1']);
                    geom_transformer_ELP.feature(wp_number).geom.create(rec_number1, 'Rectangle');
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number1).set('pos', [-D_cond_to_core_P-(count_P_L-1)*(width_Pri+D_cond_to_cond_P) -(EE_width_mid/2+D_cond_to_core_P+(count_P_L-1)*(width_Pri+D_cond_to_cond_P))]);
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number1).set('size', [EE_length+2*D_cond_to_core_P+2*(count_P_L-1)*(width_Pri+D_cond_to_cond_P) EE_width_mid+2*D_cond_to_core_P+2*(count_P_L-1)*(width_Pri+D_cond_to_cond_P)]);
                    geom_transformer_ELP.feature(wp_number).geom.run(rec_number1);
                    geom_transformer_ELP.feature(wp_number).geom.create(fil_number1,'Fillet');
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number1).selection('point').set(join([rec_number1 '(1)']),1:4);
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number1).set('radius',D_cond_to_core_P);
                    geom_transformer_ELP.feature(wp_number).geom.run(fil_number1);
                    %
                    rec_number2 = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_rec_2']);
                    fil_number2 = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_fil_2']);
                    geom_transformer_ELP.feature(wp_number).geom.create(rec_number2, 'Rectangle');
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number2).set('pos', [-D_cond_to_core_P-width_Pri-(count_P_L-1)*(width_Pri+D_cond_to_cond_P) -(EE_width_mid/2+D_cond_to_core_P+width_Pri+(count_P_L-1)*(width_Pri+D_cond_to_cond_P))]);
                    geom_transformer_ELP.feature(wp_number).geom.feature(rec_number2).set('size', [EE_length+2*D_cond_to_core_P+2*width_Pri+2*(count_P_L-1)*(width_Pri+D_cond_to_cond_P) EE_width_mid+2*D_cond_to_core_P+2*width_Pri+2*(count_P_L-1)*(width_Pri+D_cond_to_cond_P)]);
                    geom_transformer_ELP.feature(wp_number).geom.run(rec_number2);
                    geom_transformer_ELP.feature(wp_number).geom.create(fil_number2,'Fillet');
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number2).selection('point').set(join([rec_number2 '(1)']),1:4);
                    geom_transformer_ELP.feature(wp_number).geom.feature(fil_number2).set('radius',D_cond_to_core_P);
                    geom_transformer_ELP.feature(wp_number).geom.run(fil_number2);
                    %
                    diff_winding = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_diff']);
                    geom_transformer_ELP.feature(wp_number).geom.create(diff_winding,'Difference');   
                    geom_transformer_ELP.feature(wp_number).geom.feature(diff_winding).selection('input').set({fil_number2});
                    geom_transformer_ELP.feature(wp_number).geom.feature(diff_winding).selection('input2').set({fil_number1});
                    geom_transformer_ELP.feature(wp_number).geom.run(diff_winding);
                    %
                end
                extrude_winding = join(['B_' num2str(count_board) '_P_ext']);
                geom_transformer_ELP.create(extrude_winding, 'Extrude');     % add a new feature - extrude 
                geom_transformer_ELP.feature(extrude_winding).selection('input').set({wp_number});
                geom_transformer_ELP.feature(extrude_winding).set('distance', thickness_cond);
                geom_transformer_ELP.feature(extrude_winding).set('selresult', 'on');
                geom_transformer_ELP.run(extrude_winding);                   % build extrude
                count_selection_copper = count_selection_copper+1;
                %selection_name_copper = join([extrude_winding '_dom']);
                Union_selection_copper{count_selection_copper} = extrude_winding;
            end
            %
            geom_transformer_ELP.feature.create('copper_dom_selection', 'UnionSelection');
            geom_transformer_ELP.feature('copper_dom_selection').set('input', Union_selection_copper);
            geom_transformer_ELP.run('copper_dom_selection');
            comp_transformer_ELP.material('mat1').selection.named('geom1_copper_dom_selection');
            %
            comp_transformer_ELP.view('view1').set('transparency', true);
            mphgeom(model_transformer_ELP);
     %% apply physics interface
            % create internal boundary on which current excitation to be
            % applied
            %
            geom_transformer_ELP.create('wp2', 'WorkPlane');    % add a new feature - work plane 'wp1' to the geometry 'geom_transformer_ELP'
            wp2 = geom_transformer_ELP.feature('wp2');
            wp2.set('planetype', 'quick');
            wp2.set('quickplane', 'yz');
            wp2.set('quickx', 0);
            %
            %
            wp2.geom.create('partition_rec', 'Rectangle');
            wp2.geom.feature('partition_rec').set('pos', [EE_width_mid/2 -E_height_inner]);
            wp2.geom.feature('partition_rec').set('size', [(EE_width_inner-EE_width_mid)/2 2*E_height_inner]);
            wp2.geom.feature('partition_rec').set('selresult', 'on');
            wp2.geom.run('partition_rec');
            %
            geom_transformer_ELP.create('par1', 'Partition');
            geom_transformer_ELP.feature('par1').selection('input').set(Union_selection_copper);
            geom_transformer_ELP.feature('par1').set('partitionwith', 'objects');
            %geom_transformer_ELP.feature('par1').set('partitionwith', 'workplane');
            %geom_transformer_ELP.feature('par1').set('workplane', 'wp2');
            geom_transformer_ELP.feature('par1').selection('tool').set({'wp2'});
            geom_transformer_ELP.run('par1');
            %
            comp_transformer_ELP.physics.create('mf', 'InductionCurrents', 'geom1');    % create magnetic filed interface
            mf = comp_transformer_ELP.physics('mf');
            comp_transformer_ELP.physics('mf').feature('al1').setIndex('materialType', 'from_mat', 0);  % ampere's law
            %
            % assign coil to secondary windings
            for count_board = 1:1:N_B_total
                %
                coil_box_base_line_z = -E_height_inner+distance_B_to_B+(count_board-1)*displacement_B_to_B;
                coil_box_z_max = coil_box_base_line_z-thickness_cond/5;
                coil_box_z_min = coil_box_base_line_z-thickness_cond/4;
                %
                for count_S_L = 1:1:N_S_L
                    %
                    coil_box_y_max = EE_width_mid/2+D_cond_to_core_S+width_Sec/2+(count_S_L-1)*(D_cond_to_cond_S+width_Sec);
                    coil_box_y_min = EE_width_mid/2+D_cond_to_core_S+width_Sec/3+(count_S_L-1)*(D_cond_to_cond_S+width_Sec);
                    %
                    coil_bnd_selection_name_input = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_bnd_input']);
                    geom_transformer_ELP.create(coil_bnd_selection_name_input,'BoxSelection');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('entitydim', 2);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('ymin', coil_box_y_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('ymax', coil_box_y_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('xmin', -D_cond_to_core_S);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('xmax', D_cond_to_core_S);
                    %{
                    coil_bnd_selection_name_output = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_bnd_output']);
                    geom_transformer_ELP.create(coil_bnd_selection_name_output,'BoxSelection');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('entitydim', 2);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('ymax', -coil_box_y_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('ymin', -coil_box_y_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('xmin', -D_cond_to_core);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('xmax', D_cond_to_core);
                    %}
                    coil_dom_selection_name = join(['B_' num2str(count_board) '_S_' num2str(count_S_L) '_dom_sel']);
                    geom_transformer_ELP.create(coil_dom_selection_name,'BoxSelection');
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('entitydim', 3);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('ymin', coil_box_y_min);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('ymax', coil_box_y_max);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('xmin', -D_cond_to_core_S);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('xmax', D_cond_to_core_S);
                    %
                    coil_name = join(['coil_' 'B_' num2str(count_board) '_S_' num2str(count_S_L)]);
                    mf.create(coil_name, 'Coil', 3);
                    mf.feature(coil_name).set('CoilName', coil_name);
                    mf.feature(coil_name).setIndex('materialType', 'from_mat', 0);
                    mf.feature(coil_name).selection.named(join(['geom1_' coil_dom_selection_name]));
                    mf.feature(coil_name).set('ICoil', I_rms_S);
                    mf.feature(coil_name).feature('ccc1').feature('ct1').selection.named(join(['geom1_' coil_bnd_selection_name_input]));
                    %{
                    mf.feature(coil_name).feature('ccc1').create('cg1', 'CoilGround', 2);
                    mf.feature(coil_name).feature('ccc1').feature('cg1').selection.named(join(['geom1_' coil_bnd_selection_name_output]));
                    %}
                    mf.feature(coil_name).feature('ccc1').feature('ct1').set('Reverse', true);
                end
            end
            %
            % assign coil to primary windings
            for count_board = 1:1:N_B_total
                %
                coil_box_base_line_z = -E_height_inner+count_board*displacement_B_to_B;
                coil_box_z_max = coil_box_base_line_z+thickness_cond/4;
                coil_box_z_min = coil_box_base_line_z+thickness_cond/5;
                %
                for count_P_L = 1:1:N_P_L
                    %
                    coil_box_y_max = EE_width_mid/2+D_cond_to_core_P+width_Pri/2+(count_P_L-1)*(D_cond_to_cond_P+width_Pri);
                    coil_box_y_min = EE_width_mid/2+D_cond_to_core_P+width_Pri/3+(count_P_L-1)*(D_cond_to_cond_P+width_Pri);
                    %
                    coil_bnd_selection_name_input = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_bnd_input']);
                    geom_transformer_ELP.create(coil_bnd_selection_name_input,'BoxSelection');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('entitydim', 2);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('ymin', coil_box_y_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('ymax', coil_box_y_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('xmin', -D_cond_to_core_P);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_input).set('xmax', D_cond_to_core_P);
                    %{
                    coil_bnd_selection_name_output = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_bnd_output']);
                    geom_transformer_ELP.create(coil_bnd_selection_name_output,'BoxSelection');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('entitydim', 2);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('ymax', -coil_box_y_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('ymin', -coil_box_y_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('xmin', -D_cond_to_core);
                    geom_transformer_ELP.feature(coil_bnd_selection_name_output).set('xmax', D_cond_to_core);
                    %}
                    coil_dom_selection_name = join(['B_' num2str(count_board) '_P_' num2str(count_P_L) '_dom_sel']);
                    geom_transformer_ELP.create(coil_dom_selection_name,'BoxSelection');
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('entitydim', 3);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('condition', 'intersects');
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('ymin', coil_box_y_min);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('ymax', coil_box_y_max);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('zmin', coil_box_z_min);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('zmax', coil_box_z_max);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('xmin', -D_cond_to_core_P);
                    geom_transformer_ELP.feature(coil_dom_selection_name).set('xmax', D_cond_to_core_P);
                    %
                    coil_name = join(['coil_' 'B_' num2str(count_board) '_P_' num2str(count_P_L)]);
                    mf.create(coil_name, 'Coil', 3);
                    mf.feature(coil_name).set('CoilName', coil_name);
                    mf.feature(coil_name).setIndex('materialType', 'from_mat', 0);
                    mf.feature(coil_name).selection.named(join(['geom1_' coil_dom_selection_name]));
                    mf.feature(coil_name).set('ICoil', I_rms_P_COMSOL);
                    mf.feature(coil_name).feature('ccc1').feature('ct1').selection.named(join(['geom1_' coil_bnd_selection_name_input]));
                    %{
                    mf.feature(coil_name).feature('ccc1').create('cg1', 'CoilGround', 2);
                    mf.feature(coil_name).feature('ccc1').feature('cg1').selection.named(join(['geom1_' coil_bnd_selection_name_output]));
                    %}
                end
            end
            %
    %% add air
            %
            geom_transformer_ELP.create('s1','Sphere');
            geom_transformer_ELP.feature('s1').set('r',air_factor*EE_width_outer);
            geom_transformer_ELP.feature('s1').set('layer',5e-3);
            geom_transformer_ELP.feature('s1').set('pos',[EE_length/2 0 0]);
            geom_transformer_ELP.feature('s1').set('selresult', 'on');
            geom_transformer_ELP.run('s1');
            %
            %
            geom_transformer_ELP.runPre('fin');
            geom_transformer_ELP.run('fin'); 
            %
            %
            geom_transformer_ELP.create('ball_1','BallSelection');
            geom_transformer_ELP.feature('ball_1').set('entitydim', 3);
            geom_transformer_ELP.feature('ball_1').set('condition', 'intersects');
            geom_transformer_ELP.feature('ball_1').set('posx', EE_length/2);
            geom_transformer_ELP.feature('ball_1').set('posy', 0);
            geom_transformer_ELP.feature('ball_1').set('posz', 0);
            geom_transformer_ELP.feature('ball_1').set('r', EE_length/2);
            %
            geom_transformer_ELP.create('infinite_selection','DifferenceSelection');
            geom_transformer_ELP.feature('infinite_selection').set('add', {'s1'});
            geom_transformer_ELP.feature('infinite_selection').set('subtract', {'ball_1'});
            %
            comp_transformer_ELP.coordSystem.create('infinite_air', 'InfiniteElement');
            comp_transformer_ELP.coordSystem('infinite_air').set('ScalingType', 'Spherical');
            comp_transformer_ELP.coordSystem('infinite_air').selection.named('geom1_infinite_selection');
            %
            geom_transformer_ELP.create('ball_2','BallSelection');
            geom_transformer_ELP.feature('ball_2').set('entitydim', 3);
            geom_transformer_ELP.feature('ball_2').set('condition', 'intersects');
            geom_transformer_ELP.feature('ball_2').set('posx', EE_length/2);
            geom_transformer_ELP.feature('ball_2').set('posy', 0);
            geom_transformer_ELP.feature('ball_2').set('posz', E_height_outer + 5e-3);
            geom_transformer_ELP.feature('ball_2').set('r', 1e-3);
            %comp_transformer_ELP.material('mat3').selection.named('geom1_ball_2');
            geom_transformer_ELP.feature.create('Air_dom_selection', 'UnionSelection');
            geom_transformer_ELP.feature('Air_dom_selection').set('input', {'infinite_selection' 'ball_2'});
            geom_transformer_ELP.run('Air_dom_selection');
            comp_transformer_ELP.material('mat3').selection.named('geom1_Air_dom_selection');
    %% create mesh
            comp_transformer_ELP.mesh.create('mesh1');
            comp_transformer_ELP.mesh('mesh1').automatic(false);
            comp_transformer_ELP.mesh('mesh1').feature.remove('ftet1');
            comp_transformer_ELP.mesh('mesh1').create('swept_coil', 'Sweep');
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').selection.geom('geom1', 3);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').selection.named('geom1_copper_dom_selection');
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').set('facemethod', 'tri');
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').create('size1', 'Size');
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('custom', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hmaxactive', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hminactive', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hmax', maximum_size_winding);   % maximum element size for mesh on winding
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hmin', minimum_size_winding);    % minimum element size for for mesh on winding
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hgradactive', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hcurveactive', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hnarrowactive', true);
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hgrad', max_growth_rate_winding);      % maximum element growth rate for mesh on winding
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hnarrow', curvature_factor_winding);        % curvature factor for mesh on winding
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('size1').set('hcurve', resol_narrow_widning);     %resolution of narrow regions for mesh on winding
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').create('dis1', 'Distribution');
            comp_transformer_ELP.mesh('mesh1').feature('swept_coil').feature('dis1').set('numelem', number_e_face_widning);   % number of elements on face
            comp_transformer_ELP.mesh('mesh1').create('ftet1', 'FreeTet');  % mesh remaining domains
            comp_transformer_ELP.mesh('mesh1').feature('size').set('custom', true);
            comp_transformer_ELP.mesh('mesh1').feature('size').set('hmax', maximum_size_general);    % maximum element size for mesh on domains other than winding
            comp_transformer_ELP.mesh('mesh1').feature('size').set('hmin', minimum_size_general);         % minimum element size for mesh on domains other than winding
            comp_transformer_ELP.mesh('mesh1').feature('size').set('hgrad', max_growth_rate_general);       % maximum element growth rate for mesh on domains other than winding
            comp_transformer_ELP.mesh('mesh1').feature('size').set('hcurve', curvature_factor_general);     % curvature factor for mesh on domains other than winding
            comp_transformer_ELP.mesh('mesh1').feature('size').set('hnarrow', resol_narrow_general);     % resolution of narrow regions for mesh on domains other than winding
            comp_transformer_ELP.mesh('mesh1').run;
            %
    %% create study
            model_transformer_ELP.study.create('std1');
            model_transformer_ELP.study('std1').setGenConv(true);
            model_transformer_ELP.study('std1').create('ccc', 'CoilCurrentCalculation');
            model_transformer_ELP.study('std1').create('freq', 'Frequency');
            model_transformer_ELP.study('std1').feature('freq').set('punit', 'Hz');
            model_transformer_ELP.study('std1').feature('freq').set('plist', f_s_COMSOL);
            %
            model_transformer_ELP.sol.create('sol1');
            model_transformer_ELP.sol('sol1').study('std1');
            %
            model_transformer_ELP.study('std1').feature('ccc').set('notlistsolnum', 1);
            model_transformer_ELP.study('std1').feature('ccc').set('notsolnum', '1');
            model_transformer_ELP.study('std1').feature('ccc').set('listsolnum', 1);
            model_transformer_ELP.study('std1').feature('ccc').set('solnum', '1');
            model_transformer_ELP.study('std1').feature('freq').set('notlistsolnum', 1);
            model_transformer_ELP.study('std1').feature('freq').set('notsolnum', 'auto');
            model_transformer_ELP.study('std1').feature('freq').set('listsolnum', 1);
            model_transformer_ELP.study('std1').feature('freq').set('solnum', 'auto');
            %
            model_transformer_ELP.sol('sol1').create('st1', 'StudyStep');
            model_transformer_ELP.sol('sol1').feature('st1').set('study', 'std1');
            model_transformer_ELP.sol('sol1').feature('st1').set('studystep', 'ccc');
            model_transformer_ELP.sol('sol1').create('v1', 'Variables');
            model_transformer_ELP.sol('sol1').feature('v1').set('control', 'ccc');
            model_transformer_ELP.sol('sol1').create('s1', 'Stationary');
            model_transformer_ELP.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
            model_transformer_ELP.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'dDef');
            model_transformer_ELP.sol('sol1').feature('s1').feature.remove('fcDef');
            model_transformer_ELP.sol('sol1').create('su1', 'StoreSolution');
            model_transformer_ELP.sol('sol1').create('st2', 'StudyStep');
            model_transformer_ELP.sol('sol1').feature('st2').set('study', 'std1');
            model_transformer_ELP.sol('sol1').feature('st2').set('studystep', 'freq');
            model_transformer_ELP.sol('sol1').create('v2', 'Variables');
            model_transformer_ELP.sol('sol1').feature('v2').set('initmethod', 'sol');
            model_transformer_ELP.sol('sol1').feature('v2').set('initsol', 'sol1');
            model_transformer_ELP.sol('sol1').feature('v2').set('notsolmethod', 'sol');
            model_transformer_ELP.sol('sol1').feature('v2').set('notsol', 'sol1');
            model_transformer_ELP.sol('sol1').feature('v2').set('control', 'freq');
            model_transformer_ELP.sol('sol1').create('s2', 'Stationary');
            model_transformer_ELP.sol('sol1').feature('s2').create('p1', 'Parametric');
            model_transformer_ELP.sol('sol1').feature('s2').feature.remove('pDef');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('pname', {'freq'});
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('plistarr', {num2str(f_s_COMSOL)});
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('punit', {'Hz'});
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('pcontinuationmode', 'no');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('preusesol', 'auto');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('pdistrib', 'off');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('plot', 'off');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('plotgroup', 'Default');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('probesel', 'all');
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('probes', {});
            model_transformer_ELP.sol('sol1').feature('s2').feature('p1').set('control', 'freq');
            model_transformer_ELP.sol('sol1').feature('s2').set('control', 'freq');
            model_transformer_ELP.sol('sol1').feature('s2').set('linpmethod', 'sol');
            model_transformer_ELP.sol('sol1').feature('s2').set('linpsol', 'sol1');
            model_transformer_ELP.sol('sol1').feature('s2').set('linpsoluse', 'su1');
            model_transformer_ELP.sol('sol1').feature('s2').set('stol', 0.1);
            model_transformer_ELP.sol('sol1').feature('s2').create('fc1', 'FullyCoupled');
            model_transformer_ELP.sol('sol1').feature('s2').create('i1', 'Iterative');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').set('linsolver', 'bicgstab');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').set('prefuntype', 'right');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').set('rhob', 400);
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').create('mg1', 'Multigrid');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('pr').create('sv1', 'SORVector');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('pr').feature('sv1').set('prefun', 'sorvec');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('pr').feature('sv1').set('sorvecdof', {'comp1_A'});
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('po').create('sv1', 'SORVector');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('po').feature('sv1').set('prefun', 'soruvec');
            model_transformer_ELP.sol('sol1').feature('s2').feature('i1').feature('mg1').feature('po').feature('sv1').set('sorvecdof', {'comp1_A'});
            model_transformer_ELP.sol('sol1').feature('s2').feature('fc1').set('linsolver', 'i1');
            model_transformer_ELP.sol('sol1').feature('s2').feature.remove('fcDef');
            model_transformer_ELP.sol('sol1').feature('v2').set('notsolnum', 'auto');
            model_transformer_ELP.sol('sol1').feature('v2').set('notsolvertype', 'solnum');
            model_transformer_ELP.sol('sol1').feature('v2').set('notlistsolnum', {'1'});
            model_transformer_ELP.sol('sol1').feature('v2').set('notsolnum', 'auto');
            model_transformer_ELP.sol('sol1').feature('v2').set('notlistsolnum', {'1'});
            model_transformer_ELP.sol('sol1').feature('v2').set('notsolnum', 'auto');
            model_transformer_ELP.sol('sol1').feature('v2').set('control', 'freq');
            model_transformer_ELP.sol('sol1').feature('v2').set('solnum', 'auto');
            model_transformer_ELP.sol('sol1').feature('v2').set('solvertype', 'solnum');
            model_transformer_ELP.sol('sol1').feature('v2').set('listsolnum', {'1'});
            model_transformer_ELP.sol('sol1').feature('v2').set('solnum', 'auto');
            model_transformer_ELP.sol('sol1').feature('v2').set('listsolnum', {'1'});
            model_transformer_ELP.sol('sol1').feature('v2').set('solnum', 'auto');
            model_transformer_ELP.sol('sol1').attach('std1');
            %
            model_transformer_ELP.sol('sol1').runAll;
            %
            % extract resistance of secondary winding
            Winding_resistance = 0;
            for count_board = 1:1:N_B_total
                %
                for count_S_L = 1:1:N_S_L
                    %
                    coil_name = join(['coil_' 'B_' num2str(count_board) '_S_' num2str(count_S_L)]);
                    Winding_resistance = Winding_resistance+mphglobal(model_transformer_ELP,join(['mf.RCoil_' coil_name]),'dataset','dset1');
                end
            end
            Winding_Sec_resistance_COMSOL = Winding_resistance;
            Winding_Sec_loss_COMSOL = I_rms_S*I_rms_S*Winding_Sec_resistance_COMSOL;
            %
            % extract resistance of primary winding
            Winding_resistance = 0;
            for count_board = 1:1:N_B_total
                %
                %
                for count_P_L = 1:1:N_P_L
                    %
                    coil_name = join(['coil_' 'B_' num2str(count_board) '_P_' num2str(count_P_L)]);
                    Winding_resistance = Winding_resistance+mphglobal(model_transformer_ELP,join(['mf.RCoil_' coil_name]),'dataset','dset1');
                end
            end
            Winding_Pri_resistance_COMSOL = Winding_resistance;
            Winding_Pri_loss_COMSOL = I_rms_P_COMSOL*I_rms_P_COMSOL*Winding_Pri_resistance_COMSOL;
            %
            Winding_loss_COMSOL = Winding_Sec_loss_COMSOL+Winding_Pri_loss_COMSOL;
        else
            disp('wrong input');
        end