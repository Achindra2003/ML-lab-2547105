# Lab Questions & Answers Document
**India Air Quality & Crop Yield — EDA Lab**

This document directly answers the prompts and deliverables requested in the `LAB questions .docx.md` file based on the analysis performed in our Jupyter Notebooks.

---

### Task 1: Structured Data Profile & Initial Inspection
**Deliverable: Structured data profile and written concern**

*Data Profile:*
- **City AQI (`city_day.csv`)**: 29,531 rows, 16 columns. Spans 2015–2020 across 26 cities. High missingness in pollutants (Xylene missing >61%). Highly right-skewed.
- **Crop Production (`crop_production.csv`)**: 246,091 rows, 7 columns. Spans 1997–2015 across 33 states/UTs. Exceptionally clean (only ~1.5% missing Production).

*Written Concern:*
The biggest red flag is the **Temporal Mismatch**. The crop data ends in 2015, and the AQI data starts in 2015. There is only one overlapping year for direct daily comparison. Furthermore, the crop data measures Coconuts in "Nuts" and Wheat in "Tonnes", meaning we cannot simply sum production without filtering out non-weight units first.

---

### Task 2: The data has holes — and not all holes are equal
**Deliverable: Missing value treatment strategy and justification**

*Strategy:*
1. **Crop Production**: Imputed using `Area × Median Yield` for the specific crop. **Why:** A global mean gives tiny farms the same output as massive ones. Multiplying the land area by typical yield perfectly scales the filled data to the physical size of the farm.
2. **AQI (Air Quality)**: Imputed using `Linear Interpolation`. **Why:** Replacing missing days with a yearly average destroys the continuous weather trend. Interpolation draws a straight line between readings, preserving winter pollution spikes.
3. **Xylene (Pollutant)**: Dropped entirely. **Why:** Over 61% missing. Imputing would fabricate the majority of the data.

---

### Task 3: The two files disagree on how to spell "Tamil Nadu"
**Deliverable: Inconsistencies found and fixes applied**

*Inconsistencies & Fixes:*
1. **Missing State Column in AQI:** The AQI dataset only had cities. We mapped all 26 cities to their 21 standard parent states using a Python dictionary.
2. **Trailing Spaces in Crops:** We applied `.str.strip()` to state and crop names to fix invisible grouping errors (e.g., `" Rice "` vs `"Rice"`).
3. **Deduplication:** We ran `drop_duplicates()` on both files, which returned 0 exact row duplicates. The files are now structurally ready for a state-level aggregation merge.

---

### Task 4: Where do most cities actually sit on the AQI scale?
**Deliverable: Visualisation choice, justification, and observations**

*Visualisation Chosen:* A **Histogram with KDE** overlaid with Mean/Median lines, alongside a **Grouped Boxplot sorted by Median**. 
*Justification:* The histogram shows the shape of the data (where values cluster), while the sorted boxplot shows the extreme values for specific cities.

*Observations:*
1. **The Average is Unfair:** The data is heavily right-skewed. The Median is 118 (Moderate), but extreme winter outliers drag the Mean up to 164.6.
2. **Geographical Concentration:** Clean cities (like Aizawl) have tight, low distributions. The extreme outliers are entirely clustered in northern industrial/riverine plains (Delhi, Patna, Ahmedabad).

---

### Task 5: Something is making the average AQI look worse than it is
**Deliverable: Detection, treatment, and visual comparison**

*Treatment:* **IQR Winsorization (Capping at the 95th Percentile)**
*Justification:* We detected 1,512 extreme outliers (AQI > 398.0) using the IQR upper fence. Instead of deleting them, we "capped" them at 398.0. 
*Why:* If we delete outlier rows, we delete the most important winter days from our timeline, creating gaps. Capping retains the fact that it was a "very high pollution day" while stopping mathematically absurd sensor glitches (like AQI 2000) from breaking the ML models.

---

### Task 6: Is India's air getting better or worse over time?
**Deliverable: Trend visualisation and response to journalist**

*Response to the Journalist:*
Looking at the data from 2015 to 2020, urban air quality in India shows a gradual overall improvement. Average pollution levels declined from a peak mean of 191 in 2015 to a low of 111 in 2020. A noticeable decline occurs after 2018, correlating with the National Clean Air Programme (NCAP). However, the sharp drop in 2020 was heavily influenced by temporary COVID-19 lockdowns, so we must be cautious about attributing all success to long-term policy.

---

### Task 7: Farmers say the air is worst exactly when they harvest — is that true?
**Deliverable: Seasonal pattern analysis and response to NGO**

*Response to the NGO:*
The data partially supports the claim, but reveals a deeper truth. Air quality does degrade severely during the winter post-harvest season, crossing into the "Poor" category. However, the pollution peaks in December and January—long after the active October harvesting is finished. This proves that crop residue burning is NOT the sole driver of the winter crisis. The extended peak is caused by **Winter Inversion**—cold air trapping industrial and vehicular smog close to the ground. 

---

### Task 8: Can the two datasets talk to each other?
**Deliverable: Transformation strategy and multi-variable relationships**

*Transformation:* We aggregated both datasets to the **State Level**. We calculated the long-term Average AQI for each state, and the cumulative Production/Area for agriculture. We also explicitly filtered out "Coconuts" to prevent fruit counts from distorting metric ton weights.

*Interesting Relationships:*
1. **The "Fertilizer Trap" (Moderate Positive Correlation between AQI and Production):** 
   *Reason:* Highly polluted states (like Punjab and Gujarat) are massive industrial hubs. Because they are wealthy, they use immense amounts of chemical fertilizers and tractors. This makes it look like pollution causes high yields, but it's an illusion caused by economic wealth.
2. **Weak Correlation between AQI and Yield Per Hectare:**
   *Reason:* Yield is dominated by capital inputs (irrigation, soil chemistry). These inputs completely buffer the physical damage that smog does to the plants.

---

### Task 9: The minister needs to act — what do you tell her?
**Deliverable: 150-200 word briefing**

**To: The State Environment Minister**  
**Subject: Key Findings on Urban Air Quality and Agricultural Crop Yields**  

Honorable Minister,  

Ahead of your cabinet meeting, here is a concise summary of our data analysis linking urban air quality and agricultural production:

1. **Pollution Levels are Falling**: National urban air quality has improved, with average AQI dropping 42% from 191 in 2015 to 111 in 2020. Emission policies are working, though urban air remains above satisfactory limits.
2. **Severe Winter Spike Confirmed**: Air pollution is highly seasonal, peaking at a poor index of 212 in winter. This is a combined effect of agricultural crop-burning and winter weather inversion layers trapping pollutants.
3. **Agricultural Yields are Buffered**: At a state level, highly polluted states maintain high yields because advanced irrigation and fertilizer inputs offset the physical damage of pollution.

**Recommendation**: Subsidize mechanical seeders (like Happy Seeders) in agricultural hubs. This enables farmers to clear crop residues without burning, directly attacking the November pollution spike.

**Honest Limitation**: This dataset measures correlation, not direct cause. We cannot definitively prove physical plant damage, as we lack local variables like rainfall and exact fertilizer volumes.

---

### Task A: The two extremes — do they tell the same story?
*Analysis:* The extremes contradict the simple hypothesis. The cleanest states (Mizoram, Meghalaya) have very low crop yields, while the most polluted states (Gujarat, Bihar) have very high yields. This shows that **economic geography and topography** confound the relationship. Clean air states are mountainous with limited flat land, while polluted states are flat riverine plains that support intensive farming.

### Task B: Put a number on the relationship
*Analysis:* We calculated a **Pearson Correlation (r = 0.42)** between Average AQI and Log Production. This means there is a moderate positive association. However, this absolutely does not prove that pollution causes yield gains. It simply proves that agricultural powerhouse states are also highly developed and mechanized, and that shared economic activity drives both high yields and high pollution.

### Task C: One plot to rule them all
*Visualisation Choice:* The Dual-Panel Scatter Plot comparing Raw Production vs. Filtered Coconut Production against AQI.
*Caption:* "This chart demonstrates the danger of naive data aggregation. By comparing state crop production against air quality before and after filtering out non-weight units (Coconuts), it reveals how a single reporting inconsistency can completely distort national correlation models, highlighting that meticulous data cleaning is the only barrier between statistical truth and statistical fiction."
