function driftControl(pixelCount)
    condition = sum(pixelCount(106:145) > 40);
    if condition
        disp("EMERGENCY")
    end
end

