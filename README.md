# Blotify
 Blotify is an open-source image analysis tool designed for automated, grid-based quantification of dot blot assays.

Automate your dot blot analysis with smart grid detection and integrated density measurement. No more manual ROI clicks!"

**⚠️ Work in Progress (WIP):** Active development ongoing. This right now is a fully functional proof of concept. 

### **Phase 1: Stable CLI (Current Goal)**  
- [x] Automated Grid detection + density quantification  
- [ ] Add CLI arguments for input/output control  
- [ ] Export options (CSV, Excel, JSON)  

### **Phase 2: Interactive Features**  
- [ ] Web app for manual adjustments (image rotation, manual grid tweaking)  
- [ ] Real-time calculation preview  

##  How It Works  
### **Core Algorithms**  
1. **Preprocessing**  
   - *Inversion + Background Subtraction* (Top-Hat/Rolling Ball)  
   - *Adaptive Thresholding* - optional  (Otsu/Gaussian) to isolate dot shapes.  

2. **Grid Detection**  
   - *Projection Profiling*: Find best seperating lines via row/column intensity peaks.  
   - *Morphological Cleanup - optional * (Erosion/Dilation) to filter artifacts.  

3. **Quantification**  
   - *Integrated Density*: Sums pixel intensities per cell (background-subtracted).  

4. **Visualization**
  - Overlays grids/calculations for validation.  
