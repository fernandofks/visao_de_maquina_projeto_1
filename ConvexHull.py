import cv2
  
# Read the image
image = cv2.imread("NOK_borda\Fig_NOK_12.jpg",cv2.IMREAD_GRAYSCALE)
returns,thresh=cv2.threshold(image,105,255,cv2.THRESH_BINARY)

# Find the contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
for i in range(len(contours)):
    if len(contours[i]) >= 800:
        cv2.drawContours(image,[contours[i]],-1,(255,0,0),2)
        
        # Get the convex hull of the contour
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(image,hull,-1,(0,255,0),2)
        
        # Check if the contour is convex
        if len(hull) == len(contours[i]):
            print(i,'The contour is convex')
        else:
            print(i,'The contour is not convex')
        
    
cv2.imshow('Convexity Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()