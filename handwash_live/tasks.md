pending tasks
=============
1. [ ] Collect more data to train handwash model.
    - Write script that records and saves the video if model is unable to classify the steps correctly.
2. [ ] Keep handwash model active even if hand detection model was unable to detect hand in a frame. But with one condition which is if hand detection model is unable to detect hand in frames for `t` seconds then deactivate the handwash model.