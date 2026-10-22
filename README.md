# SafariSeg

เว็บแอปตรวจจับและแบ่งส่วน ช้าง ม้าลาย ยีราฟ ด้วย YOLO26n-seg (polygon)
โมเดลรันบนเบราว์เซอร์ทั้งหมดผ่าน onnxruntime-web เป็น static site ไม่มี backend

## โครงสร้าง
```
index.html      หน้าเว็บ
style.css
app.js          preprocess -> ONNX inference -> decode -> mask -> polygon -> วาดผล
model/best.onnx Polygon weight ที่ export จาก best.pt
model/meta.json ชื่อ label, ผลการเทรน, รายชื่อภาพตัวอย่าง
assets/         กราฟผลการเทรนและภาพ predict
samples/        ภาพตัวอย่างจากชุด test
render.yaml     ตั้งค่า Render (Static Site)
```

## ทดสอบในเครื่อง
```
cd webapp
python -m http.server 8000
```
เปิด http://localhost:8000

## Deploy ขึ้น Render
1. push โฟลเดอร์นี้ขึ้น GitHub repo ใหม่
2. Render > New > Static Site > เลือก repo
3. Build Command: เว้นว่าง, Publish Directory: `.`
4. กด Create Static Site รอสักครู่จะได้ลิงก์ `https://<ชื่อ>.onrender.com`
