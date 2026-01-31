clear

%% Lendo data-set para terino da rede
itens = load('data_set.mat').tabela1;

input = [itens.Queda, itens.Vazao]';
output = [itens.Rendimento , itens.Potencia, itens.DistribuidorD, itens.RotorD]';

%% Definições da rede para treinamento
net = feedforwardnet(40,'trainlm');

net.trainParam.epochs = 500;

net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio   = 20/100;
net.divideParam.testRatio  = 20/100;

%% Treino da rede neural
net = train(net, input, output, "useParallel", "yes",...
                'showResources', 'yes',...
                'useGPU', 'no'...
                );
%% Usando a rede para prever os dados
predicted = net(input)
%% Inicio do cálculo de erros
MAE = sqrt(mean(output'-predicted').^2)
RMSE = sqrt(mean((output'-predicted').^2))
R = []
for i = 1:size(output,1)
    mdl = fitlm(output(1,:)',predicted(1,:)')
    R = [R,mdl.Rsquared.Adjusted]
end
Error = table(MAE', RMSE',R')
%% Erros calculados e armazenados, removendo então as variaveis inuteis
clear MAE RMSE R mdl i

fnew = figure;
plot(1:50,output(1,1:50),'r--*', 1:50,predicted(1,1:50),'b--o')
plot(1:50,output(2,1:50),'r--*', 1:50,predicted(2,1:50),'b--o')
plot(1:50,output(3,1:50),'r--*', 1:50,predicted(3,1:50),'b--o')
plot(1:50,output(4,1:50),'r--*', 1:50,predicted(4,1:50),'b--o')
