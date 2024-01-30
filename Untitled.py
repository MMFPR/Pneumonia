# التحكم في التحذيرات ومنع ظهورها أثناء التنفيذ يتم تعطيل جميع التحذيرات
import warnings
warnings.filterwarnings('ignore')
# استيراد المكتبات المستخدمة لبناء نموذج التعلم العميق VGG16
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
# استيراد المكتبات لتحميل نموذج مدرب سابقاً لتحميل ومعالجة الصور
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# معالجة البيانات
import numpy as np
# تحديد حجم الصورة المستخدمة في النموذج
IMAGE_SIZE = [224, 224]
#تحديد مسارات مجلدات الاختبار والتدريب
train_path = 'Datasets/train'
test_path = 'Datasets/test'
#إنشاء نموذج VGG16 مع الشكل المحدد والأوزان المدربة مسبقا وعدم تضمين الطبقة العلوية
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
#تجميد جميع الطبقات في نموذج VGG16 لمنع تدريبها خلال التدريب الجديد
for layer in vgg.layers:
    layer.trainable = False
#الحصول على قائمة بالفئات الموجودة في مجلد التدريب.
folders = glob('Datasets/train/*')
#إضافة طبقة التسطيح للمخرجات الخاصة بنموذج VGG16
x = Flatten()(vgg.output)
#إضافة طبقة كاملة متصلة للتصنيف مع استخدام وحدات softmax
prediction = Dense(units=len(folders), activation="softmax")(x)
#بناء النموذج النهائي باستخدام الطبقات المضافة والطبقات المجمدة من VGG16
model = Model(inputs=vgg.input, outputs=prediction)
#طباعة ملخص للنموذج لعرض هيكله
model.summary()
#تكوين عملية التدريب بتحديد خسارة التصنيف الهرمي ومعدل التحديث المستخدم ومقاييس الأداء
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#تكوين مولد البيانات للتدريب مع عمليات التحويل المتنوعة
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#تكوين مولد البيانات للاختبار مع تحويل البيانات إلى نطاق [0,1]
test_datagen = ImageDataGenerator(rescale=1./255)
#إنشاء مولد لبيانات التدريب باستخدام مولد البيانات وتحديد حجم الصور وحجم الدفعة batch size ونمط الفئات
training_set = train_datagen.flow_from_directory("Datasets/train", target_size=(224, 224), batch_size=10, class_mode="categorical")
#بيانات الاختبار
test_set = test_datagen.flow_from_directory("Datasets/test", target_size=(224, 224), batch_size=10, class_mode="categorical")
#تدريب النموذج باستخدام مولد البيانات لعدد محدد من الحلقات epochs
r = model.fit_generator(training_set, validation_data=test_set, epochs=1,steps_per_epoch=len(training_set), validation_steps=len(test_set))

#حفظ النموذج
model.save("chest_xray1.h5")
#تحميل النموذج المدرب
model = load_model("chest_xray1.h5")
# تحميل صورة
img = image.load_img('C:\\Users\\Majid\\Desktop\\project_5\\Datasets\\val\\NORMAL\\NORMAL2-IM-1427-0001.jpeg', target_size=(224, 224))
#تحويل الصورة إلى تنسيق قابل للتحليل من قبل النموذج
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# تنصيف الصورة بناء على النموذج
classes = model.predict(img_data)
result = classes[0][0]

#اختبار النتيجة واظاهر حالة الصورة
if result > 0.5:
    print("النتيجة طبيعية")
else:
    print("الشخص مصاب بالتهاب الرئة")

