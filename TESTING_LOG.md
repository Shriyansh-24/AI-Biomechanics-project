001 [28/2/2026]- Encountered and resolved an AttributeError by managing environment dependencies. Chose to utilize MediaPipe 0.10.14 to maintain compatibility with legacy Pose-estimation architectures while evaluating the migration path to the MediaPipe Tasks API.

002 [1/3/2026] - Valgus warning firing repeatedly even though my knees were not actually caving inward. The valgus algorithm works by comparing the knee's horizontal (x-axis) position against the midpoint of the hip and ankle. This is valid from the front where the camera can see left-right movement clearly. However from the side view, all three landmarks (hip, knee, ankle) have nearly identical x-coordinates because they are all the same horizontal distance from the camera. 
      Fix: Made seperate modes for side view and front view to check knee valgus and depth seperately.
      
003 [1/3/2026] - Shows Risk > 40 and depth issues while standing normally.
      Fix: Added a boolean to check whether person is standing still or in the middle of rep.

004 - Added feature to seperate csv tables and risk screenshots of each run
005 - Program showing knees caving inwards when actually caving outwards. When knees are caving inwards it is undetected.
      Fix: Fixed the camera mirror by reversing the values
Add: varus detection — outward knee deviation now flagged alongside valgus

