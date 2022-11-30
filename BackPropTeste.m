clear;
%Entrada
input = [0 0; 0 1; 1 0; 1 1];
numlin = length (input(:,1));
numcol = length (input(1,:));

%Saída
output = [0; 1; 1; 0];
tam1 = length (output(:,1));
tam2 = length (output(1,:));

%Número de neurônios da camada oculta
nn = 1 ;
maxn = nn + numcol + tam2;

%Bias
bias = -1*ones(1,maxn);

%Coeficiente de Aprendizado
na = 1;

%Número de iterações (épocas)
epochs = 10^4;

%Pesos
valor = max(numcol+1,nn+1);
pesos = ones(maxn,valor);
%pesos = [ 0 0 0; 0.1 0.1 0.1; 0.1 0.1 0.1; 0.1 0.1 0.1];

%Função Custo
custo = zeros(epochs,1);
for i = 1:epochs
    delta = ones(maxn,1);
    y = zeros (tam1,tam2);
    
    
    for j = 1:numlin
        
        %Etapa feedfoward - Camada de Entrada
        e = zeros(numcol,1);
        
        for k = 1:numcol
            e(k) = input(j,k);
        end
        
        %Etapa feedfoward - Camada Oculta
        z = zeros(nn,1);
        
        for k = 1:nn
            temp = 0;
            for l = 2:numcol+1
                temp = temp + e(l-1)*pesos(k+numcol,l);
            end
            temp = temp + bias(1,2+k)*pesos(k+numcol,1);
            z(k) = sigma(temp);
        end
                
        %Etapa feedfoward - Camada de Saída
        soma = 0;
        
        for k = 1:tam2
            for l = 1:nn
                soma = z(l)*pesos(nn+numcol+k,l+1);
            end
            yy = bias(1,nn+numcol+k)*pesos(nn+numcol+k,1) + soma;
            y(j,k) = sigma(yy);
        end
        
        %Etapa backpropagation
        
        custo(i) = immse(output,y);
        
        
        for k = 1:tam2
            delta(nn+k) = y(j,k)*(1 - y(j,k))*(output(j,k) - y(j,k));
        end
        
        for k = 1:nn
            temp = 0;
            for l = 1:tam2
                temp = temp + pesos(nn+l,k+1)*delta(nn+l);
            end
            delta(k) = z(k)*(1 - z(k))*temp/tam2;
        end
        
        for k = 1:nn
            for l = 1:numcol
                pesos(numcol+k,l+1) = pesos(numcol+k,l+1) + na*e(l)*delta(k);
            end
            pesos(numcol+k,1) = pesos(numcol+k,1) + na*bias(1,numcol+k)*delta(k);
        end
        
        for k = 1:tam2
            for l = 1:nn
                pesos(numcol+nn+k,l+1) = pesos(numcol+nn+k,l+1) + na*z(l)*delta(nn+k);
            end
            pesos(numcol+nn+k,1) = pesos(numcol+nn+k,1) + na*bias(1,numcol+nn+k)*delta(nn+k);
        end
        
    end
end
%plot(custo)
% disp (y)
% disp (pesos)            
% classes = vec2ind (y');
% saida = vec2ind (output);
% plot(saida,'or')
% hold
% plot(classes,'xg')