%% original

function [lossGenerator, lossDiscriminator] = ganLoss(probReal, probGenerated)
%GANLOSS Compute the total loss of the GAN.

% Copyright 2020 The MathWorks, Inc.

% Calculate losses for discriminator network
lossGenerated = -mean(log(1 - probGenerated));
lossReal = -mean(log(probReal));

% Combine losses for discriminator network
lossDiscriminator = lossReal + lossGenerated;

% Calculate loss for generator network
lossGenerator = -mean(log(probGenerated));%non-saturating gan loss
end
%% WGAN

% function [lossGenerator, lossDiscriminator] = ganLoss(probReal, probGenerated)
% %GANLOSS Compute the total loss of the GAN.
% 
% % Copyright 2020 The MathWorks, Inc.
% 
% % Calculate losses for discriminator network
% lossGenerated = mean(probGenerated);
% lossReal = mean(probReal);
% 
% % Combine losses for discriminator network
% lossDiscriminator = lossReal - lossGenerated;
% 
% % Calculate loss for generator network
% lossGenerator = mean(probGenerated);%non-saturating gan loss
% end


%% LSGAN LOSS

% function [lossGenerator, lossDiscriminator] = ganLoss(probReal, probGenerated)
% %GANLOSS Compute the total loss of the GAN.
% 
% % Copyright 2020 The MathWorks, Inc.
% 
% % Calculate losses for discriminator network
% lossReal = mean((probReal-1).^2);
% lossGenerated = mean(probGenerated.^2);
% 
% % Combine losses for discriminator network
% lossDiscriminator = lossReal - lossGenerated;
% 
% % Calculate loss for generator network
% lossGenerator = mean((probGenerated-1).^2);
% end

