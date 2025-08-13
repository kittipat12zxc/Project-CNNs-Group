# ผลการดำเนินงาน  
**นาย นพดล ตั้งปัญญาผล**  
รหัสนักศึกษา: 6611011660027  

---

## 1. Dataset ที่ใช้
ใช้ Dataset อาหารไทย โดยมีคลาสทั้งหมด 8 ประเภท:
- แกงเขียวหวาน  
- ข้าวผัด  
- ข้าวซอย  
- แกงมัสมั่น  
- ข้าวผัดกระเพรา  
- ผัดไทย  
- ส้มตำ  
- ต้มยำ  

Dataset นี้ใช้สำหรับการ **Training** และ **Testing** โมเดล
![data]("D:\Github\งานกลุ่มจารืยเเนน\Project-CNNs-Group\images\Screenshot 2025-08-13 165050.png")

---

## 2. Data Preparation
- รูปภาพอาหารไทยได้จาก:
  - อินเทอร์เน็ต
  - กล้องมือถือถ่ายเอง
- แบ่งข้อมูล:
  - Training: ~400 รูป
  - Testing: ~100 รูป  
  - อัตราส่วน 90/10
- ทำ Data Augmentation:
  - หมุนภาพ
  - ปรับสีภาพ
  - Zoom
  - Blur  

<p align="center">
  <img src="images/augmentation-example.png" alt="Data Augmentation Example" width="400"/>
</p>

---

## 3. อธิบาย Model ที่เลือกใช้
เลือกใช้ **ResNet-18** 2 แบบ:
1. ResNet-18 ที่สร้างเองจาก PyTorch (Custom Layer ได้)
2. ResNet-18 สำเร็จรูปจาก `torchvision` (เลือกโหลด pretrained weights และ fine-tune ได้)

### โหมดการทำงานของ ResNet-18
1. **Fine-tuning**  
   - ใช้โมเดล pretrained (ImageNet)
   - ปรับแต่งทุก layer
   - เปลี่ยน classification head ให้รองรับ 8 classes
2. **Feature Extraction**  
   - ใช้ pretrained
   - Freeze hidden layers
   - ปรับเฉพาะ classification head
3. **Train from Scratch**  
   - ไม่มี pretrained weights
   - เทรนทุก layer ใหม่

---

## 4. การ Training / Testing
- จำนวน Epochs: `25`
- Batch size: `64`
- Learning rate: `0.002`

**ผลลัพธ์ที่ดีที่สุด**  
- Training Acc: **97.688%**  
- Validation Acc: **95.125%**  
- Training Loss: **0.113**  
- Validation Loss: **0.183**  

<p align="center">
  <img src="images/accuracy-graph.png" alt="Accuracy Graph" width="400"/>
  <br>
  <em>กราฟ Accuracy: เพิ่มขึ้นต่อเนื่อง ไม่มี overfitting</em>
</p>

<p align="center">
  <img src="images/loss-graph.png" alt="Loss Graph" width="400"/>
  <br>
  <em>กราฟ Loss: ลดลงต่อเนื่อง</em>
</p>

---

## 5. สรุปผลการจำแนก
จำแนกได้ทั้งหมด 8 คลาส (ตามหัวข้อ Dataset)  
- **Pad Krapraw:** 71% (ดีที่สุด)  
- **Tom Yum:** 6% (ต่ำสุด)  

---

## 6. ปัญหา / อุปสรรค และข้อเสนอแนะ
1. Overfitting:
   - เกิดจาก learning rate สูงเกินไป
   - แก้โดยเพิ่ม augmentation และปรับ hyperparameters
2. ปัญหา dependencies:
   - Version ของ library ไม่ตรงกัน
   - แก้โดยติดตั้งเวอร์ชันที่รองรับ:
     ```bash
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
     ```
3. ความแม่นยำของบางคลาสต่ำ:
   - โดยเฉพาะ Tom Yum, Som Tum, Khao Phat  
   - แนะนำเพิ่ม dataset ให้มากขึ้น

---

