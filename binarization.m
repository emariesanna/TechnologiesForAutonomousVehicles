function outputImg = binarization(inputImg)
    arrayImg = inputImg(:);
    maxVal = max(arrayImg);
    minVal = min(arrayImg);
    
    newThreshold = double((maxVal + minVal)/2);
    threshold = 0.0;
    
    while(abs(newThreshold - threshold) > 0.1)
        
        threshold = newThreshold;
        upperArray = (arrayImg(arrayImg > threshold));
        lowerArray = (arrayImg(arrayImg <= threshold));
        
        meanUpper = mean(upperArray);
        meanLower = mean(lowerArray);
        
        newThreshold = (meanUpper + meanLower)/2;
    end
    
    outputImg = inputImg > newThreshold;

end

