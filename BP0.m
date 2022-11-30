%Entrada
input = [0 0; 0 1; 1 0; 1 1];

%Saída
output = [0; 1; 1; 0];

%Bias
bias = [-1 -1 -1];

%Coeficiente de Aprendizado
n = 1;

%Número de iterações (épocas)
epochs = 100000;

%Pesos
%pesos = [0 0 0; 0.1 0.1 0.1; 0.1 0.1 0.1];
pesos = rand(3+nn,valor);

for i = 1:epochs
    
    out = zeros (4,1);
    numln = length (input(:,1));
    
    for j = 1:numln
        
        %Etapa feedfoward
        
        H1 = bias(1,1)*pesos(1,1) + input(j,1)*pesos(1,2) + input(j,2)*pesos(1,3);
        x2(1) = sigma(H1);
        
        H2 = bias(1,2)*pesos(2,1) + input(j,1)*pesos(2,2) + input(j,2)*pesos(2,3);
        x2(2) = sigma(H2); 
        
        x3_1 = bias(1,3)*pesos(3,1) + x2(1)*pesos(3,2) + x2(2)*pesos(3,3);
        out(j) = sigma(x3_1);
        
        delta3_1 = out(j)*(1 - out(j))*(output(j) - out(j));
        delta2_1 = x2(1)*(1 - x2(1))*pesos(3,2)*delta3_1;
        delta2_2 = x2(2)*(1 - x2(2))*pesos(3,3)*delta3_1;                                                                        
        
        for k = 1:3
            if k == 1
                pesos(1,k) = pesos(1,k) + n*bias(1,1)*delta2_1;
                pesos(2,k) = pesos(2,k) + n*bias(1,2)*delta2_2;
                pesos(3,k) = pesos(3,k) + n*bias(1,3)*delta3_1;
            else
                pesos(1,k) = pesos(1,k) + n*input(j,1)*delta2_1;
                pesos(2,k) = pesos(2,k) + n*input(j,2)*delta2_2;
                pesos(3,k) = pesos(3,k) + n*x2(k-1)*delta3_1;
            end
        end
    end
end
disp (out)
pesos             
function y = sigma(x)
y = 1./(1 + exp(-x));
end