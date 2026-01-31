<h1> Generation of Turbine Hill Chart </h1>

![ANN](https://user-images.githubusercontent.com/39101353/123632549-74ff2a80-d7ee-11eb-8278-cffd86993561.png)

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=Fellipe0/Hill-Chart-A&project=code.mlx&file=HillChart)

```matlab
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
```

```
Starting parallel pool (parpool) using the 'local' profile ...
Connected to the parallel pool (number of workers: 12).
 
Computing Resources:
Parallel Workers:
  Worker 1 on Felipe_PC, MEX on PCWIN64
  Worker 2 on Felipe_PC, MEX on PCWIN64
  Worker 3 on Felipe_PC, MEX on PCWIN64
  Worker 4 on Felipe_PC, MEX on PCWIN64
  Worker 5 on Felipe_PC, MEX on PCWIN64
  Worker 6 on Felipe_PC, MEX on PCWIN64
  Worker 7 on Felipe_PC, MEX on PCWIN64
  Worker 8 on Felipe_PC, MEX on PCWIN64
  Worker 9 on Felipe_PC, MEX on PCWIN64
  Worker 10 on Felipe_PC, MEX on PCWIN64
  Worker 11 on Felipe_PC, MEX on PCWIN64
  Worker 12 on Felipe_PC, MEX on PCWIN64
```

```matlab
%% Usando a rede para prever os dados
predicted = net(input)
```

```text:Output
predicted = 4x1243
   56.4138   58.6678   61.0201   63.3958   65.8019   68.1478   70.4262   72.5396   74.4826   76.1942   77.7101   79.0214   80.1738   81.1679   82.0189   82.7961   83.5552   84.3494   85.1375   85.8655   86.4601   86.9167   87.2496   87.4951   87.6638   87.7665   87.8125   87.8174   87.7909   87.7341   87.6441   87.5143   87.3446   87.1327   86.8861   86.6048   86.2997   85.9703   85.6276   85.2688   84.9024   84.5208   84.1273   83.7070   83.2589   82.7656   82.2268   81.6527   81.0479   80.4501
    3.3752    3.9602    4.5986    5.2787    6.0104    6.7719    7.5655    8.3620    9.1641    9.9514   10.7420   11.5275   12.3250   13.1214   13.9032   14.6904   15.4826   16.3045   17.1333   17.9656   18.7680   19.5492   20.2976   21.0343   21.7494   22.4592   23.1522   23.8437   24.5213   25.1967   25.8551   26.5067   27.1374   27.7585   28.3582   28.9486   29.5197   30.0843   30.6334   31.1792   31.7117   32.2401   32.7508   33.2485   33.7183   34.1655   34.5856   34.9789   35.3579   35.7149
   26.9805   29.5941   32.1554   34.5850   36.9085   39.0725   41.1207   43.0243   44.8329   46.5247   48.1415   49.6584   51.1072   52.4742   53.7497   54.9640   56.1023   57.1950   58.2266   59.2201   60.1596   61.0687   61.9371   62.7866   63.6030   64.4022   65.1699   65.9230   66.6493   67.3651   68.0584   68.7441   69.4110   70.0736   70.7210   71.3672   72.0006   72.6338   73.2541   73.8724   74.4757   75.0747   75.6579   76.2373   76.8024   77.3640   77.9154   78.4504   78.9821   79.5113
  -16.9839  -16.2621  -15.5265  -14.7936  -14.0514  -13.3157  -12.5741  -11.8423  -11.1082  -10.3871   -9.6661   -8.9592   -8.2526   -7.5529   -6.8653   -6.1739   -5.4886   -4.7943   -4.1052   -3.4108   -2.7271   -2.0422   -1.3690   -0.6939   -0.0290    0.6388    1.2971    1.9584    2.6094    3.2615    3.9010    4.5389    5.1619    5.7810    6.3836    6.9806    7.5602    8.1331    8.6884    9.2366    9.7670   10.2900   10.7958   11.2942   11.7761   12.2507   12.7128   13.1569   13.5894   13.9990
```

```matlab
%% Inicio do cálculo de erros
MAE = sqrt(mean(output'-predicted').^2)
RMSE = sqrt(mean((output'-predicted').^2))
R = []

for i = 1:size(output,1)
    mdl = fitlm(output(1,:)',predicted(1,:)')
    R = [R,mdl.Rsquared.Adjusted]
end
```

```text:Output
mdl = 
Linear regression model:
    y ~ 1 + x1

Estimated Coefficients:
                   Estimate        SE        tStat      pValue 
                   ________    __________    ______    ________

    (Intercept)    0.04425       0.024169    1.8308    0.067364
    x1             0.99953     0.00026704      3743           0

Number of observations: 1243, Error degrees of freedom: 1241
Root Mean Squared Error: 0.0559
R-squared: 1,  Adjusted R-Squared: 1
F-statistic vs. constant model: 1.4e+07, p-value = 0
R = 0.9999
mdl = 
Linear regression model:
    y ~ 1 + x1

Estimated Coefficients:
                   Estimate        SE        tStat      pValue 
                   ________    __________    ______    ________

    (Intercept)    0.04425       0.024169    1.8308    0.067364
    x1             0.99953     0.00026704      3743           0

Number of observations: 1243, Error degrees of freedom: 1241
Root Mean Squared Error: 0.0559
R-squared: 1,  Adjusted R-Squared: 1
F-statistic vs. constant model: 1.4e+07, p-value = 0
R = 1x2
    0.9999    0.9999

mdl = 
Linear regression model:
    y ~ 1 + x1

Estimated Coefficients:
                   Estimate        SE        tStat      pValue 
                   ________    __________    ______    ________

    (Intercept)    0.04425       0.024169    1.8308    0.067364
    x1             0.99953     0.00026704      3743           0

Number of observations: 1243, Error degrees of freedom: 1241
Root Mean Squared Error: 0.0559
R-squared: 1,  Adjusted R-Squared: 1
F-statistic vs. constant model: 1.4e+07, p-value = 0
R = 1x3
    0.9999    0.9999    0.9999

mdl = 
Linear regression model:
    y ~ 1 + x1

Estimated Coefficients:
                   Estimate        SE        tStat      pValue 
                   ________    __________    ______    ________

    (Intercept)    0.04425       0.024169    1.8308    0.067364
    x1             0.99953     0.00026704      3743           0

Number of observations: 1243, Error degrees of freedom: 1241
Root Mean Squared Error: 0.0559
R-squared: 1,  Adjusted R-Squared: 1
F-statistic vs. constant model: 1.4e+07, p-value = 0
R = 1x4
    0.9999    0.9999    0.9999    0.9999

```

```matlab
Error = table(MAE', RMSE',R')
```

| |Var1|Var2|Var3|
|:--:|:--:|:--:|:--:|
|1|0.0020|0.0559|0.9999|
|2|0.0003|0.0218|0.9999|
|3|0.0016|0.0471|0.9999|
|4|0.0004|0.0325|0.9999|

```matlab
%% Erros calculados e armazenados, removendo então as variaveis inuteis
clear MAE RMSE R mdl i

fnew = figure;
plot(1:50,output(1,1:50),'r--*', 1:50,predicted(1,1:50),'b--o')
```

![ANN](https://github.com/Fellipe0/Hill-Chart-A.I/blob/main/code_media/figure_0.png)

```matlab
plot(1:50,output(2,1:50),'r--*', 1:50,predicted(2,1:50),'b--o')
```


![ANN](https://github.com/Fellipe0/Hill-Chart-A.I/blob/main/code_media/figure_1.png)

```matlab
plot(1:50,output(3,1:50),'r--*', 1:50,predicted(3,1:50),'b--o')
```


![ANN](https://github.com/Fellipe0/Hill-Chart-A.I/blob/main/code_media/figure_2.png)

```matlab
plot(1:50,output(4,1:50),'r--*', 1:50,predicted(4,1:50),'b--o')
```


![ANN](https://github.com/Fellipe0/Hill-Chart-A.I/blob/main/code_media/figure_3.png)

## Requirements:
<table>
  <tr>
    <td>Platform</td>
    <td>Version</td>
  </tr>
  <tr>
    <td>Matlab</td>
    <td>2020a</td>
  </tr>
</table>
