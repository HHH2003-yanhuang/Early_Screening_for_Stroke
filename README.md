# AI-Based_Early_Screening_for_Stroke_in_Primary_Healthcare
Due to the high incidence of cerebral stroke in my hometown, I developed an intelligent early screening model for cerebral stroke that can be used in homes and primary care facilities based on resampling, cost-sensitive learning, and ensemble learning.

Using this model, you can input the following personal information to obtain both the predicted probability of stroke risk and a binary diagnosis result.

Attention，please! The model supports a customizable classification threshold (0 ≤ threshold ≤ 1) , enabling users to balance between minimizing missed diagnoses (life safety) and reducing false positives (medical costs and psychological burden) based on individual value preferences.


Input:

gender: Male / Female

age: Age (integer)

hypertension: Hypertension (0 = No, 1 = Yes)

heart_disease: Heart disease (0 = No, 1 = Yes)

ever_married: Ever married (Yes / No)

work_type: Work type (Children / Private / Never_worked / Self-employed / Govt_job)

Residence_type: Residence type (Urban / Rural)

avg_glucose_level: Average glucose level (unit: mg/dL)

bmi: Body Mass Index (BMI)

smoking_status: Smoking status (Smokes / Formerly smoked / Never smoked)

threshold=x (0 ≤ x ≤ 1)

<img width="570" height="117" alt="image" src="https://github.com/user-attachments/assets/d9016eb0-0f2b-4486-8219-8ce8955a49c8" />

<img width="633" height="422" alt="image" src="https://github.com/user-attachments/assets/a963caa7-4a6b-433f-948a-e6c151a2c8ad" />


Output：

Predicted stroke risk probability: x.xx%

Stroke diagnosis: Yes / No

<img width="472" height="77" alt="image" src="https://github.com/user-attachments/assets/19070b4c-09c8-40b2-b2e1-cc4cf7121653" />
