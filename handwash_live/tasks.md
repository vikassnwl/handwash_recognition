pending tasks
=============
1. [ ] Collect more data to train handwash model.
    - Write script that records and saves the video if model is unable to classify the steps correctly.
2. [ ] Keep handwash model active even if hand detection model was unable to detect hand in a frame. But with one condition which is if hand detection model is unable to detect hand in frames for `t` seconds then deactivate the handwash model.
3. [ ] Test the previous model on merged dataset's test set and compare with new model after training. Assign the predicted label to the entire video based on the max labels assigned to the frames and then calculate accuracy based on videos not frames.