clear;

%Entrada
input = [0 0; 0 1; 1 0; 1 1];
length (input(:,1));

%Saída
output = [0; 1; 1; 0];
tam = length (output(:,1));

%Número de neurônios da camada oculta
nn = 1;

delta = ones(nn+3,1);

%Bias
bias = ones(nn+3,1);

%Coeficiente de Aprendizado
na = 0.1;

%Número de iterações (épocas)
epochs = 10^5;

%Pesos
valor = max(3,nn+1);
pesos = rand(3+nn,valor);
%pesos = [ 0 0 0; 0.1 0.1 0.1; 0.1 0.1 0.1; 0.1 0.1 1];

for i = 1:epochs
    
    y = zeros (tam,1);
    numln = length (input(:,1));
    custo = 0;
    for j = 1:numln
          
        %Etapa feedfoward
        
        z = zeros(nn);
        
        for k = 1:nn
            temp = bias(2+k) + input(j,1)*pesos(k+2,1) + input(j,2)*pesos(k+2,2);
            z(k) = sigma(temp);
        end
                        
        somaz = 0;
        for k = 1:nn
            somaz = somaz + z(k)*pesos(3+nn,k);
        end
        
        yy = bias(3+nn) + somaz;
        y(j) = sigma(yy);
        
        
        custo = custo + y(j) - output(j);
        %Etapa backpropagation
        
        somaz = 0;
        somapeso = 0;
        somain = input(j,1)+input(j,2);
        
        for k = 1:nn
            somaz = somaz + z(k);
            somapeso = somapeso + pesos(4,k);
        end
        
        delta(nn+3) = (output(j)-y(j))*sigma(yy)*(1-sigma(yy));
        
        for k = 1:nn
            delta(k) = sigma(temp)*(1-sigma(temp))*delta(nn+3)*pesos(nn+3,k);
        end
        
        for k = 1:nn
            pesos(nn+3,k) = pesos(nn+3,k) + na*delta(nn+3)*z(k);
        end
        
        bias(nn+3) = bias(nn+3) + na*delta(nn+3);
        
        
        for k = 1:nn
            bias(k+2) = bias(k+2) + na*delta(k);
            pesos(k+2,1) = pesos(k+2,1) + na*delta(k)*input(j,1);
            pesos(k+2,2) = pesos(k+2,2) + na*delta(k)*input(j,2);
        end
       
    end
    
    end
disp(pesos)
disp(y)


        