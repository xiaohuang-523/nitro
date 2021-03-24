%% Spline 3D curve fitting tutorial
npts = 13;
t = linspace(0,8*pi,npts);
z = linspace(-1,1,npts);
omz = sqrt(1-z.^2);
xyz = [cos(t).*omz; sin(t).*omz; z];
plot3(xyz(1,:),xyz(2,:),xyz(3,:),'ro','LineWidth',2);
text(xyz(1,:),xyz(2,:),xyz(3,:),[repmat('  ',npts,1), num2str((1:npts)')])
ax = gca;
ax.XTick = [];
ax.YTick = [];
ax.ZTick = [];
%box on

hold on
fnplt(cscvn(xyz(:,[1:end 1])),'r',2)
hold off
%% Spline curve fitting working data
% pp form reference: https://www.mathworks.com/help/matlab/ref/unmkpp.html
% https://www.mathworks.com/help/matlab/ref/mkpp.html
spline_pc = csvread('G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_points.csv');
x = spline_pc(:,1);
y = spline_pc(:,2);
z = spline_pc(:,3);
xyz = spline_pc.'
plot3(x,y,z,'ro','LineWidth',2);
hold on
fnplt(cscvn(xyz(:,[1:end 1])),'r',2)
%cscvn(xyz(:,[1:end 1]))
cscvn(xyz(:,[1:end 1]))
%cscvn(xyz)
[breaks,coefs,L,order,dim] = unmkpp(cscvn(xyz))
%breaks = cscvn(xyz).breaks;
%coef = cscvn(xyz).coefs
%writematrix(breaks, 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_breaks.csv')
%writematrix(coef, 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_coefficients.csv')
