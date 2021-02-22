clear; close all; clc; clf;
load carsmall
format long
diary vanillaPerceptronResults.txt
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
w = zeros(1, size(x,2));

%Perceptron algorithm
for i=1:nIter
    for j=1:m
        if(w*x(j,:)'<=0)
           w = w + y(j)*x(j,:);
        end
    end
end

%The performance is tested
yTest = x*w';
yTest(yTest>0)=1;
yTest(yTest<0)=-1;
nWrongClassif = sum(abs(yTest-y));
nGoodClassif = m - nWrongClassif;
errorAverage = sum(abs(yTest-y))/m;

%Decision boundary is plotted
if(n==2)
    xx = linspace(min(x(:,1)),max(x(:,1)));
    yy = linspace(min(x(:,2)),max(x(:,2)));
    [xx, yy]=meshgrid(xx,yy);
    z= w(1)*xx+w(2)*yy;
    contour(xx,yy,z,[0,0], 'g'); 
    legend('X Feture','Y Feature', 'Decision Boundary' ,'Location','Best');
end


time = datestr(clock,'YYYY/mm/dd HH:MM:SS:FFF');
fprintf("***************************************************\n");
fprintf('%23s\n', time);
fprintf("Vanilla perceptron algorithm has been trained in: %f seconds.\n", toc);
fprintf("Good classified examples: %d out of %d\n", nGoodClassif, m);
fprintf("Wrong classified examples: %d out of %d\n\n\n", nWrongClassif, m);

diary off;