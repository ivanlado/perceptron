clear; close all; clc; clf;
load carsmall
format long
diary averagedPerceptronResults.txt
tic

%Hyperparameters
nIter = 1e1; %Number of iterations
minNumGoodClassif = 0;

%Data is loaded
data = load("perceptron_data.csv");

%Data is split
m = length(data);
n = size(data, 2)-1;
x = data(:,1:n);
y = data(:,n+1);

%X data is centred
x = x - mean(x);

%Y data is treated
y(y>0)=1;
y(y<=0)=-1;

%Data is plot using a scattered plot if and only if has only 2 features
if(n==2)
    hold on;
    gscatter(x(:,1), x(:,2), y, ['r', 'b']);
end

%Parameters are initialized
k=1;
w = zeros(1, n);
u = zeros(1, n);
duration = 0;
for i = 1:nIter
   for j = 1:m
       if ((x(j,:)*w(k,:)')*y(j))<=0
           u = u + y(j)* w * duration;
           w = w + y(j)*x(j,:)*(duration>=minNumGoodClassif);
           duration = 0;
       else
           duration = duration + 1;
       end
   end
   u = u + y(j)* w * duration; 
   if(i==nIter)
       break;
   end
   if(duration==m)
       w =  (rand(1,n)*2)-1;
   end
   
end


%Predictions are made taking into account the duration for each weight
yTilde=x*w';
yTilde(yTilde>0)=1;
yTilde(yTilde<0)=-1;

%Errors are
nWrongClassif = sum((yTilde-y)~=0);
nGoodClassif = m - nWrongClassif;

time = datestr(clock,'YYYY/mm/dd HH:MM:SS:FFF');
fprintf("***************************************************\n");
fprintf('%23s\n', time);
fprintf("Averaged perceptron algorithm has been trained in: %f seconds.\n", toc);
fprintf("Good classified examples: %d out of %d\n", nGoodClassif, m);
fprintf("Wrong classified examples: %d out of %d\n\n\n", nWrongClassif, m);

diary off;