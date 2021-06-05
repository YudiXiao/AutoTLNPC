function [V_gain] = Vgain(N_p,N_s,PS,D_loss)
%The function V_gain calculates the voltage gain of the converter. To
%ensure the desired output voltage can be maintained.

    V_gain = 0.5 * (1 - PS - D_loss) * N_s / N_p;

end

