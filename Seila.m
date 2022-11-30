x = IrisInput;
t = IrisTargets;

net = feedforwardnet(10);
net.trainParam.epochs = 1000000;
net.trainParam.goal = 0.2;
net.trainFcn = 'trainscg';
net.layers{1}.transferFcn = 'purelin';
net.layers{2}.transferFcn = 'tansig';

net = train(net,x,t);
y = net(x);
perf = perform(net,t,y);
w1 = net.IW{1,1};
w2 = net.LW{2,1};
b1 = net.b{1,1};
b2 = net.b{2,1};

% y1 = round(y);
% plotconfusion(t,y1);