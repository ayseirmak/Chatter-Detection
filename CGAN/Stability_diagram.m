%% SEC 0 - Identification and Clear All Data

% Stability Diagram
% Author: Doc. Dr. H. Ozgur Unver
% Student: Hayriye AKOZ
% Last Update: November, 2021

clc;          %clear command window
clear;        %clear all variables in workspace
close all;    %close all open window

%% SEC 1 - Organizing Data Locations
%**************************************************************************

global Drive
global datapath
global tablepath

Drive = 'D:\'; 
Expset = 'Turning_Measurement\';

datapath = [Drive Expset 'Measurement\'];  
tablepath = [Drive Expset 'Measurement\'];

% Getting cutting information from "Measurement" file

filelisting = dir(datapath);
FileNames =  struct2cell(filelisting) ;
FileNames = FileNames(1, 1:end);
FileNames = natsortfiles(FileNames);
FileNames = FileNames(endsWith(FileNames, 'tdms'));

table = readtable(fullfile(tablepath,'turning_datav2.xlsx'));
VariableNames = { 'CutNo', 'rpm', 'doc', 'feed', 'd0', 'de'};
% doc (mm), feed (mm/rev), d0 - initial diameter (mm), de - end diameter (mm)
table.Properties.VariableNames = VariableNames;

%% SEC 2 - System Parameters (1DoF)
%**************************************************************************

% Turning Insert: WNMG080408-NM4 WSM20

% rake angle
alpha = -6*pi/180;                 % rad 
% chip thickness before machining
t0 = 0.02;                         % mm  
% chip thickness after machining 
tc = 0.025;                        % mm  
% chip thickness ratio
r = t0/tc;                        
% shear plane angle
phi = atan((r*cos(alpha))/(1-(r*sin(alpha)))); % rad
phi = rad2deg(phi);               % deg
alpha = rad2deg(alpha);           % deg
% friction angle from Merchant's Circle
beta_friction= 90+alpha-(2*phi);  % (deg) 
beta = (90-(beta_friction-alpha))*pi/180; % (rad)
%beta = deg2rad(55);

% Define cutting force coefficients

Ks = 850e6;                       % N/m^2 
hm = t0;  						  % mm 


% Define modal parameters for u1 direction (Repeat parameters in create_signal() function)

%Kater-Turret boşluğu 0 mm - short 
% m = 0.308;                        % kg
% k = 8.6729e07;                        % N/m
% zeta = 0.68/100;
% %wn = sqrt(k/m);                    % rad/s
% wn = 2672.55;
% c = 2*m*wn*zeta;                   % N-s/m

% % %Kater-Turret boşluğu 14 mm - long%bu
m = 0.311;                        % kg
k = 5.113e07;                          % N/m
zeta = 0.48/100;
wn = sqrt(k/m);                    % rad/s
c = 2*m*wn*zeta;                   % N-s/m  


%% SEC 3 - Numerical Time Domain Simulation of Turning
%**************************************************************************

% Simulation Parameters 

fn = sqrt(k/m)/2/pi;               % Hz
dt = 1/(50*fn);                    % s 
% number of revolutions
num_rev = 100;    

% Boundary Condition = same as xlim and ylim at SEC-8!!!

% rpm=(2000:50:4000);                      %spindle speed            
% doc=(4:0.2:10);                          %depth of cut

rpm=(3200:50:4300);                      %spindle speed        %burası degisti    
doc=(1:0.2:8.5);                          %depth of cut

[b, omega ,Numberoftotalpoint]=chatterarray(rpm, doc);

disp(sprintf('%s Start',datestr(now)))

parfor cut = (1:1:length(omega))

% number of steps per revolution
steps_rev = round(1/(dt*omega(cut)/60));     

[Acc, Disp, time, Fs] = createsignal(steps_rev, num_rev, hm, dt, Ks, b(cut), omega(cut), beta);

rms_vector(cut) =  rms(Acc);

if mod(cut,100)==0
    fprintf('%s Cut# %d \n',datestr(now), cut   );
    fprintf('%s Time Steps Per Spindle Revolution # %d \n',datestr(now), steps_rev);
end
end 

 fprintf('%s End\n',datestr(now))
 beep
 
 %% SEC 4 - Create RMS Map
 
 scalecons = max(rms_vector);

% cutparams = readmatrix([Drive 'TLAlexnetData\Simulation\Params_5mm.txt']); %AlexNet parameters

rms_vector2 = normalize(rms_vector,'scale' , scalecons);
rms_arr_mat = reshape(rms_vector2, length(rpm),length(doc))';

[m_rpm, m_doc] = meshgrid(rpm,doc);
m_docf = flip(m_doc,1);
rms_arr_matf = flip(rms_arr_mat,1);

figure(1);
surf(m_rpm,m_doc,rms_arr_mat);
colorbar;
view(0,90);

figure(2);
contourf(m_rpm,m_doc,rms_arr_mat,'LevelList', 0:0.005:1);
colormap  jet; 
contourf(m_rpm,m_doc,rms_arr_mat);
colormap  jet(100);
xlabel('Rotational speed (RPM)');
ylabel('Depth of cut (mm)');
colorbar;
hold on;

 
%% SEC 5 - System Parameters( Analytical Solution )

% Define FRF

w = (0:0.5:5000*2*pi);  % frequency, rad/s
r1 = w/wn;
FRF_real = 1/k*(1-r1.^2)./((1-r1.^2).^2 + (2*zeta*r1).^2);
FRF_imag = 1/k*(-2*zeta*r1)./((1-r1.^2).^2 + (2*zeta*r1).^2);

%% SEC 6 - Oriented FRF ( Analytical Solution )

% Directional orientation factor

mu = cos(beta);
FRF_real_orient = mu*FRF_real;
FRF_imag_orient = mu*FRF_imag;

%% SEC 7 - Determine valid chatter frequency range

% Determine valid chatter frequency range

index = find(FRF_real_orient < 0);
FRF_real_orient = FRF_real_orient(index);
FRF_imag_orient = FRF_imag_orient(index);
w = w(index);

%% SEC 8 - Calculate blim, epsilon and spindle speed

% Calculate blim

blim = -1./(2*Ks*FRF_real_orient);  % m
blim = blim*1e3;        % convert to mm

% Calculate epsilon

epsilon = zeros(1, length(FRF_imag_orient));
for cnt = 1:length(FRF_imag_orient)
    if FRF_imag_orient(cnt) < 0
        epsilon(cnt) = 2*pi - 2*atan(abs(FRF_real_orient(cnt)/FRF_imag_orient(cnt)));
    else
        epsilon(cnt) = pi - 2*atan(abs(FRF_imag_orient(cnt)/FRF_real_orient(cnt)));
    end
end

N = 150;

for i = 1:(N+1) 
omegaa(i,:) = w./(2*pi)./((i-1) + epsilon/2/pi);   % rps
omegaa(i,:) = omegaa(i,:).*60;                     % rpm
end

% Plot Results

figure(2);

for i = 1:(N+1)
plot(omegaa(i,:), blim, 'w','Linewidth',0.5)%cizgiler
hold on
end
grid minor
set(gca, 'XMinorGrid','on', 'YMinorGrid','on');
xticks(1000:50:2000);
yticks(3200:100:4300);
set(gca,'FontSize', 12)
xlabel('\Omega (rpm)')
ylabel('b_{lim} (mm)')

%
xlim([2500 4500])
ylim([0 12])


% %% SEC 9 - Sampling Points
% 
txt = {'Cut1','Cut2','Cut3','Cut4','Cut5','Cut6','Cut7','Cut8','Cut9',...
         'Cut10','Cut11','Cut12','Cut13','Cut14','Cut15'};
   
%figure(2);

for cnt = 1: length(FileNames)
    
    plot(table{cnt,2}, table{cnt,3},'o','MarkerSize',8,...
                   'MarkerEdgeColor','yellow','MarkerFaceColor','yellow');
    text( table{cnt,2}, table{cnt,3}, [ '  ', txt{cnt}], 'Color','yellow','FontSize',12);
    hold on
    
end


%% Functions

% chatterarray() 

function [b_arr omega_arr Numberoftotalpoint]=chatterarray(rpm, doc)

sizeofcolumn=length(doc);
sizeofrows=length(rpm);
Numberoftotalpoint=sizeofcolumn*sizeofrows;
omega_arr=rpm;

    for n=1:(sizeofcolumn-1)
    omega_arr=[omega_arr rpm];         
    end

b_arr=doc;
b_arr=zeros;
b2_array=ones(1,sizeofrows);
doc  = doc';

    for n=1:sizeofcolumn
    b_arr= [b_arr b2_array*doc(n,1)];   
    end
    
b_arr=b_arr(2:end);                     % mm
b_arr = b_arr .*10^-3.;                 % m

end

% createsignal()

function [Acc, Disp, time, Fs] = createsignal(steps_rev, num_rev, hm, dt, Ks, bs, omega, beta)


%Kater-Turret boşluğu 0 mm - short 
% m1 = 0.308;                        % kg
% k1 = 8.6729e07;                      % N/m
% zeta1 = 0.68/100;
% wn1 = 2672.55                    % rad/s
% c1 = 2*m1*wn1*zeta1;                   % N-s/m


% % %Kater-Turret boşluğu 14 mm - long
m1 = 0.311;                        % kg
k1 = 5.113e07;                          % N/m
zeta1 = 0.48/100;
wn1 = sqrt(k1/m1);                    % rad/s
c1 = 2*m1*wn1*zeta1;                   % N-s/m 


% Initial Conditions

u1 = 0; velocity1 = 0;  				% zero initial conditions

total_steps = num_rev*(steps_rev + 1);	% total steps

        
% Set initial surface for one revolution 
for n = 1:(steps_rev) 
	y(n) = 0;
end                             
	
% Simulation begins here
for n = (steps_rev + 1):total_steps
   ymin = hm + y(n-steps_rev);          % instantaneous depth of cut     
		
   for cnt = 1:(num_rev-1)              % Check all preceeding passes for lowest cut 
      if n > cnt*steps_rev
         ymin = hm + y(n-steps_rev);
         y1 = cnt*hm + y(n-cnt*steps_rev);
         if y1 < ymin
            ymin = y1;
         end
      end
   end
    
   F = Ks*bs*(ymin - y(n-1));
   if F < 0                             % no cutting
      F = 0;
      y(n-1) = ymin;
   end
   Fn = F*cos(beta);
   % Force(n) = F;                      % N
      
   % Perform Euler integrations 
   
   % u1 direction

   accu1 = (Fn - c1*velocity1 - k1*u1)/m1;
   velocity1 = velocity1 + accu1*dt;
   u1 = u1 + velocity1*dt;
   
   acc_u1(n)=accu1;
   disp_u1(n)=u1;
  

   % Normal direction
   
   y(n) = u1;  % m
   t(n) = (n - steps_rev - 1)*dt;       % s
   
end  % end of simulation for loop 
   
Acc = acc_u1;
Disp = disp_u1;
time = (num_rev)*60./omega;
Fs = length(Acc)./time; 

%disp(sprintf('b = %d,  omg=%d Fs=%d  t=%d\n',b, omega, Fs, time));

end



