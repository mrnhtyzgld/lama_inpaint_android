package com.example.cpponnxrunner

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.Switch
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.cpponnxrunner.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.util.*


class MainActivity : AppCompatActivity() {

    private val MODEL_ASSET_PATH = "lama_fp32.onnx"
    private val IMAGE_INPUT_PATH = "images/input_image.png"
    private val MASK_INPUT_PATH  = "images/dilated_mask.png"
    private val OUTPUT_PATH      = "output/output_image.png"

    private lateinit var binding: ActivityMainBinding
    private var ort_session: Long = -1
    private val PICK_IMAGE = 1000
    private val CAPTURE_IMAGE = 2000
    private val CAMERA_PERMISSION_CODE = 8

    // Mevcut UI’daki custom-class switch vs. dokunmuyoruz.
    private var samplesClassA = 0
    private var nameClassA: String = "A"
    private var samplesClassB = 0
    private var nameClassB: String = "B"
    private var samplesClassX = 0
    private var nameClassX: String = "X"
    private var samplesClassY = 0
    private var nameClassY: String = "Y"
    private val prepackedDefaultLabels: Array<String> = arrayOf("dog", "cat", "elephant", "cow")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // --- Assets -> Cache kopyalama ---
        // SessionCache, modeli cacheDir/inference.onnx bekliyor. O yüzden hedef adını özellikle 'inference.onnx' veriyoruz.
        copyAssetToCacheDir(MODEL_ASSET_PATH, "inference.onnx")     // => $cacheDir/inference.onnx
        copyFileOrDir("images")                                     // => $cacheDir/images/* (mask ve örnek input için, istersen)

        // Ort session oluştur (cache path veriliyor; model path'i SessionCache içinde sabitlenmiş)
        ort_session = createSession("$cacheDir")

        val inferButton: Button = findViewById(R.id.infer_button)
        inferButton.setOnClickListener(onInferenceButtonClickedListener)

        // Home screen
        binding.statusMessage.text = "cpponnxrunner"
    }

    private val onInferenceButtonClickedListener: View.OnClickListener = View.OnClickListener {
        val cameraSetting: Switch = findViewById(R.id.camera_setting)
        if (cameraSetting.isChecked) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
            } else {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, CAPTURE_IMAGE)
            }
        } else {
            val intent = Intent()
            intent.type = "image/*"
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE)
        }
    }

    private fun disableButtons() {
        binding.classA.isEnabled = false
        binding.classB.isEnabled = false
        binding.classX.isEnabled = false
        binding.classY.isEnabled = false
        binding.inferButton.isEnabled = false
    }

    private fun enableButtons() {
        if (binding.customClassSetting.isChecked) {
            binding.classA.isEnabled = true
            binding.classB.isEnabled = true
            binding.classX.isEnabled = true
            binding.classY.isEnabled = true
        }
        binding.inferButton.isEnabled = true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String?>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show()
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, CAPTURE_IMAGE)
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != RESULT_OK) return

        lateinit var srcBitMap: Bitmap
        if (requestCode == PICK_IMAGE) {
            val uri = data?.data ?: return
            srcBitMap = bitmapFromUri(uri, contentResolver)
        } else if (requestCode == CAPTURE_IMAGE) {
            srcBitMap = data?.extras?.get("data") as? Bitmap ?: return
        } else {
            return
        }

        // === img2img için giriş/çıkış boyutlarını sabitle (modeline göre güncellemek serbest) ===
        val batchSize = 1
        val channels = 3
        val width = 512
        val height = 512

        // --- Görüntüyü modele uygun boyuta getir ---
        val bitmapResized: Bitmap = processBitmap(srcBitMap)
        binding.inputImage.setImageBitmap(bitmapResized)

        // --- Image tensor (1x3xHxW, float32, NCHW) ---
        val imgData = FloatBuffer.allocate(batchSize * channels * width * height)
        imgData.rewind()
        processImage(bitmapResized, imgData, 0) // mevcut fonksiyonunu kullanıyoruz (0..1 normalize ediyorsa öyle kalsın)
        imgData.rewind()

        // --- Mask tensor (1x1xHxW, float32, NCHW) ---
        // Maskeyi assets'ten al (SABİT PATH), modele uygun boyuta getir ve 0..1 aralığında doldur.
        val maskBitmapSrc = loadBitmapFromAsset(MASK_INPUT_PATH)
        val maskBitmap = Bitmap.createScaledBitmap(maskBitmapSrc, width, height, true)
        val maskData = FloatBuffer.allocate(batchSize * 1 * width * height)
        maskData.rewind()
        processMask(maskBitmap, maskData) // 1 kanal doldur
        maskData.rewind()

        // --- JNI inference (img2img dönüşü: H*W*3 uzunlukta [0..255] float RGB interleaved) ---
        val outRgbFloats: FloatArray = performInference(
            ort_session,
            imgData.array(),
            maskData.array(),
            batchSize,
            channels,
            width,
            height
        )

        // --- Çıktıyı Bitmap'e çevir ---
        val outBitmap = floatsToBitmapRGB8(outRgbFloats, width, height)

        // --- İsteğe bağlı: cache'e kaydet ---
        val savedPath = saveBitmapToCache(outBitmap, OUTPUT_PATH)

        // --- UI gösterimi ---
        // Layout'ta outputImage ImageView'in yoksa, statusMessage ile bilgi ver.
        try {
            binding.outputImage.setImageBitmap(outBitmap)
            binding.statusMessage.text = "Output rendered."
        } catch (_: Throwable) {
            binding.statusMessage.text = "Output saved to: $savedPath"
        }

        if (requestCode == CAPTURE_IMAGE) {
            binding.cameraSetting.isChecked = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        releaseSession(ort_session)
    }

    // =========================
    // Helpers (assets/cache IO)
    // =========================

    private fun mkCacheDir(cacheFileName: String) {
        val dirs = cacheFileName.split("/")
        var extendedCacheDir = "$cacheDir"
        for (index in 0..dirs.size - 2) {
            val myDir = File(extendedCacheDir, dirs[index])
            if (!myDir.exists()) myDir.mkdir()
            extendedCacheDir = "$extendedCacheDir/${dirs[index]}"
        }
    }

    // asset -> cache (hedefte farklı isim verebilirsin)
    private fun copyAssetToCacheDir(assetFileName: String, cacheFileName: String): String {
        mkCacheDir(cacheFileName)
        val f = File("$cacheDir/$cacheFileName")
        if (!f.exists()) {
            try {
                val modelFile = assets.open(assetFileName)
                val size: Int = modelFile.available()
                val buffer = ByteArray(size)
                modelFile.read(buffer)
                modelFile.close()
                val fos = FileOutputStream(f)
                fos.write(buffer)
                fos.close()
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
        }


        return f.path
    }

    // Bir klasörü (ve altını) cache'e kopyalar (relatif yapıyı korur)
    private fun copyFileOrDir(path: String): String {
        val assetManager = assets
        try {
            val list: Array<String>? = assetManager.list(path)
            if (list == null || list.isEmpty()) {
                // asset is a file
                copyAssetToCacheDir(path, path)
            } else {
                // asset is a dir. loop over dir and copy all files or sub dirs to cache dir
                for (name in list) {
                    val p = if (path.isEmpty()) "" else "$path/"
                    copyFileOrDir(p + name)
                }
            }
        } catch (ex: IOException) {
            Log.e("cpponnxrunner", "I/O Exception", ex)
        }
        return "$cacheDir/$path"
    }

    // =========================
    // Helpers (bitmap <-> tensor)
    // =========================

    // assets/<relativePath> -> Bitmap
    private fun loadBitmapFromAsset(relativePath: String): Bitmap {
        assets.open(relativePath).use { inp ->
            return BitmapFactory.decodeStream(inp)
        }
    }

    // Maskeyi 1 kanallı float (NCHW: 1x1xHxW) olarak doldurur.
    // Basitçe gri (R kanalı) alıp 0..1'e ölçekliyoruz. (Model ihtiyacına göre değiştirebilirsin.)
    private fun processMask(maskBitmap: Bitmap, out: FloatBuffer) {
        val w = maskBitmap.width
        val h = maskBitmap.height
        val pixels = IntArray(w * h)
        maskBitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        // NCHW sırada yaz: önce tüm HxW piksel (tek kanal)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val c = pixels[y * w + x]
                val r = (c shr 16) and 0xFF
                // 0..255 -> 0..1
                out.put(r / 255f)
            }
        }
    }

    // Çıkan float RGB (0..255 interleaved) -> Bitmap
    private fun floatsToBitmapRGB8(rgb: FloatArray, width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px = IntArray(width * height)
        var i = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = clampToByte(rgb[i++])
                val g = clampToByte(rgb[i++])
                val b = clampToByte(rgb[i++])
                px[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)
        return bmp
    }

    private fun clampToByte(v: Float): Int {
        var x = v
        if (x < 0f) x = 0f
        if (x > 255f) x = 255f
        return (x + 0.5f).toInt()
    }

    private fun saveBitmapToCache(bitmap: Bitmap, relativePath: String): String {
        mkCacheDir(relativePath)
        val outFile = File("$cacheDir/$relativePath")
        FileOutputStream(outFile).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }
        return outFile.absolutePath
    }

    // =========================
    // JNI köprüleri
    // =========================

    external fun createSession(cacheDirPath: String): Long
    external fun releaseSession(session: Long)

    external fun performInference(
        session: Long,
        imageBuffer: FloatArray,  // 1x3xHxW
        maskBuffer: FloatArray,   // 1x1xHxW
        batchSize: Int,
        channels: Int,
        frameCols: Int,
        frameRows: Int
    ): FloatArray

    companion object {
        init {
            System.loadLibrary("cpponnxrunner")
        }
    }

    // =========================
    // NOT: Aşağıdaki yardımcılar projende zaten var kabul edildi.
    // - bitmapFromUri(uri, contentResolver)
    // - processBitmap(src: Bitmap) : Bitmap              // boyutlandırma / center-crop vb.
    // - processImage(bitmap: Bitmap, buf: FloatBuffer, i:Int)  // 1x3xHxW NCHW float doldurur
    // Bunlar projende yoksa, mevcutlarını kullan veya ekle.
    // =========================

    // Eğer kendi URI loader'ını kullanmak istersen minimal örnek:
    @Suppress("unused")
    private fun bitmapFromUriMinimal(uri: Uri): Bitmap {
        contentResolver.openInputStream(uri).use { input ->
            return BitmapFactory.decodeStream(input!!)
        }
    }
}