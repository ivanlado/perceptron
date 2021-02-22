clear; close all; clc; clf;
load carsmall
format long
diary votedPerceptronResults.txt
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
wRecord = zeros(1, n);
durationRecord = zeros(1);
duration = 0;
for i = 1:nIter
   for j = 1:m
       if ((x(j,:)*wRecord(k,:)')*y(j))<=0
           wRecord(k+1,:) = wRecord(k,:) + y(j)*x(j,:);
           durationRecord(k, :) = duration;
           k = k + 1;
           duration = 0;
       else
           duration = duration + 1;
       end
    end
   
   %Once the iteration has ended, it's as if prediction had failed:
   %a new w is calculated...
   durationRecord(k, :) = duration;
   
   %...unless the max num of iterations is reached
   if(i==nIter)
       break;
   end
   
   %Weights vector is prepared for a new one
   wRecord(k+1,:) = wRecord(k,:);
   k = k + 1;
   duration = 0;
   
   %If in the previous iterations, all training examples has been
   %correctly classified, the weight vector is randomly initialised,
   %so as to find the weights that minimize the errors.
   if(durationRecord(k-1, :)==m)
     wRecord(k,:) =  (rand(1,n)*2)-1;
   end
   
end


%Weights which don't meet the minimum number of good classifications
%requirement are deleted
wRecord = wRecord(find(durationRecord>minNumGoodClassif),:);
durationRecord = durationRecord(durationRecord>minNumGoodClassif);

%Accuracy is measured
xw = x*wRecord';
xw(xw>0)=1;
xw(xw<0)=-1;

%Predictions are made taking into account the duration for each weight
yTilde=(xw)*durationRecord;
yTilde(yTilde>0)=1;
yTilde(yTilde<0)=-1;

%Errors are
nWrongClassif = sum((yTilde-y)~=0);
nGoodClassif = m - nWrongClassif;

time = datestr(clock,'YYYY/mm/dd HH:MM:SS:FFF');
fprintf("***************************************************\n");
fprintf('%23s\n', time);
fprintf("Voted perceptron algorithm has been trained in: %f seconds.\n", toc);
fprintf("Good classified examples: %d out of %d\n", nGoodClassif, m);
fprintf("Wrong classified examples: %d out of %d\n\n\n", nWrongClassif, m);

diary off;