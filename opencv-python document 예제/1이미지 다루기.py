import cv2

filename = "../res/1.jpg"

original = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
unchange = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

cv2.imshow('Original', original)
cv2.imshow('Gray', gray)
cv2.imshow('Unchange', unchange)

if cv2.waitKey(0) == ord('r'):
    cv2.imwrite("test.png", gray)
cv2.destroyAllWindows()
