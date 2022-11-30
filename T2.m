clear;
load Iris-Targets.mat
load Iris-Inputs.mat

%Entrada
input = IrisInput';
numlin = length (input(:,1));
numcol = length (input(1,:));

%Saída
output = IrisTargets';
tam1 = length (output(:,1));
tam2 = length (output(1,:));

%Número de neurônios da camada oculta
nn = 1;

delta = ones(nn+numcol+1,1);

%Bias
bias = ones(1,nn+numcol+tam2);

%Coeficiente de Aprendizado
na = 0.1;

%Número de iterações (épocas)
epochs = 10^4;

%Pesos
valor = max(numcol+1,nn+1);
pesos = ones(nn+numcol+tam2,valor);
%pesos = [ 0 0 0; 0.1 0.1 0.1; 0.1 0.1 0.1; 0.1 0.1 0.1];

%Função Custo
custo = zeros(epochs,1);
for i = 1:epochs
    
    y = zeros (tam1,tam2);
    
    
    for j = 1:numlin
        
        %Etapa feedfoward - Camada de Entrada
        e = zeros(numcol,1);
        
        for k = 1:numcol
            e(k) = sigma(input(j,k));
        end
        
        %Etapa feedfoward - Camada Oculta
        z = zeros(nn,1);
        
        for k = 1:nn
            temp = 0;
            for l = 2:numcol+1
                temp = temp + e(l-1)*pesos(k+numcol,l);
            end
            temp = temp + bias(1,2+k);
            z(k) = sigma(temp);
        end
                
        %Etapa feedfoward - Camada de Saída
        soma = 0;
        
        for k = 1:tam2
            for l = 1:nn
                soma = z(l)*pesos(nn+numcol+k,l+1);
            end
            yy = bias(1,nn+numcol+k) + soma;
            y(j,k) = sigma(yy);
        end
        
        %Etapa backpropagation
        
        custo(i) = custo(i) + (1/numlin)*(output(j) - y(j))^2;
        
        delta(nn+numcol+tam2) = y(j)*(1 - y(j))*(output(j) - y(j));
        
        for k = 1:nn
            delta(k) = z(k)*(1 - z(k))*pesos(nn+numcol+tam2,k+1)*delta(nn+numcol+tam2);
        end
        
        for k = 1:nn
            for l = 2:numcol+1
                pesos(numcol+k,l) = pesos(numcol+k,l) + na*e(l-1)*delta(k);
            end
        end
        
        for k = 1:nn
            pesos(nn+numcol+tam2,k+1) = pesos(nn+numcol+tam2,k+1) + na*z(k)*delta(nn+numcol+tam2);
        end
    end
end
plot(custo)

%disp (y)
%disp (pesos)            
classes = vec2ind (y');
saida = vec2ind (IrisTargets);
plot(saida,'or')
hold
plot(classes,'xg')