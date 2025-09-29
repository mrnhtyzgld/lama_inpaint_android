package com.example.cpponnxrunner

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import java.util.concurrent.Executors // ADDED
import android.os.Handler           // ADDED
import android.os.Looper           // ADDED
import java.util.Locale            // ADDED

// TODO check optimization flags
// TODO use NNAPI accelerator
// TODO use GPU device
// TODO port your own ONNX model

class MainActivity : AppCompatActivity() {

    private val MODEL_ASSET_PATH = "lama_fp32.onnx"
    private val SAMPLE_IMAGE_ASSET = "images/input_image.jpg"
    private val SAMPLE_MASK_ASSET = "images/dilated_mask.png"
    private val OUTPUT_IMAGE_PATH = "output/output_image.png"

    private lateinit var binding: ActivityMainBinding
    private val PICK_IMAGE = 1000
    private val CAPTURE_IMAGE = 2000
    private val CAMERA_PERMISSION_CODE = 8

    // simple background executor and main-thread handler
    private val bg = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i("OpenCV", "OpenCV version = ${cvVersion()}")
        Toast.makeText(this, "OpenCV version: ${cvVersion()}", Toast.LENGTH_LONG).show()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // --- Copy Assets -> Cache ---
        copyAssetToCacheDir(MODEL_ASSET_PATH, "inference.onnx")     // => $cacheDir/inference.onnx
        copyFileOrDir("images")                                     // => $cacheDir/images/* (for sample input & mask)

        val copiedDir = File(cacheDir, "images")
        Log.i("cpponnxrunner", "images dir=${copiedDir.absolutePath} list=${copiedDir.list()?.toList()}")

        // Create ORT session in background; measure duration; show English toasts
        val modelPath = "$cacheDir/inference.onnx"
        mainHandler.post {
            Toast.makeText(this, "Loading model…", Toast.LENGTH_SHORT).show()
        }
        val t0Load = SystemClock.elapsedRealtime()
        bg.execute {
            try {
                createSession(modelPath)
                val dtMs = SystemClock.elapsedRealtime() - t0Load
                val dtSec = dtMs / 1000.0
                mainHandler.post {
                    Toast.makeText(this, String.format(Locale.US, "Model loaded (%.2f s)", dtSec), Toast.LENGTH_LONG).show()
                }
            } catch (e: Throwable) {
                Log.e("cpponnxrunner", "createSession failed", e)
                mainHandler.post {
                    Toast.makeText(this, "Model failed to load: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        val inferButton: Button = findViewById(R.id.infer_button)
        inferButton.setOnClickListener(onInferenceButtonClickedListener)

        // Home screen status
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
            startActivityForResult(Intent.createChooser(intent, "Select Image"), PICK_IMAGE)
        }
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

        try {
            // 1) Input image -> ByteArray
            val imageBytes: ByteArray = when (requestCode) {
                PICK_IMAGE -> {
                    val uri = data?.data ?: return
                    contentResolver.openInputStream(uri).use { inp ->
                        inp?.readBytes() ?: return
                    }
                }
                CAPTURE_IMAGE -> {
                    // Camera intent returns a thumbnail; compress to PNG and convert to bytes
                    val bmp = data?.extras?.get("data") as? Bitmap ?: return
                    java.io.ByteArrayOutputStream().use { bos ->
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, bos)
                        bos.toByteArray()
                    }
                }
                else -> return
            }

            // bytes-based infer (assets -> ByteArray -> JNI -> bytes -> cache) ---
            try {
                val maskBytes: ByteArray = assets.open(SAMPLE_MASK_ASSET).use { it.readBytes() }
                val imageBytes: ByteArray = assets.open(SAMPLE_IMAGE_ASSET).use { it.readBytes() }

                // Run inference in background; report duration in seconds
                mainHandler.post {
                    Toast.makeText(this, "Inference started…", Toast.LENGTH_SHORT).show()
                    binding.statusMessage.text = "Running inference…"
                }
                val t0Infer = SystemClock.elapsedRealtime()
                bg.execute {
                    try {
                        val outBytes: ByteArray = inferFromBytes(imageBytes, maskBytes)
                        val outPath = writeBytesToCache(OUTPUT_IMAGE_PATH, outBytes)
                        val dtMs = SystemClock.elapsedRealtime() - t0Infer
                        val dtSec = dtMs / 1000.0

                        // UI update (main thread)
                        mainHandler.post {
                            Log.i("cpponnxrunner", "Output saved to: $outPath")
                            val outBitmap = BitmapFactory.decodeByteArray(outBytes, 0, outBytes.size)
                            try {
                                binding.outputImage.setImageBitmap(outBitmap)
                                binding.statusMessage.text = String.format(Locale.US, "Output rendered (%.2f s)", dtSec)
                            } catch (_: Throwable) {
                                binding.statusMessage.text = String.format(Locale.US, "Output saved to: %s (%.2f s)", outPath, dtSec)
                            }
                            val inBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                            binding.inputImage.setImageBitmap(inBitmap)

                // Keep the switch on when using the camera
                            if (requestCode == CAPTURE_IMAGE) {
                                binding.cameraSetting.isChecked = true
                            }
                            Toast.makeText(this, String.format(Locale.US, "Inference finished (%.2f s)", dtSec), Toast.LENGTH_LONG).show()
                        }
                    } catch (e: Throwable) {
                        Log.e("cpponnxrunner", "bytes infer failed", e)
                        mainHandler.post {
                            binding.statusMessage.text = "Inference error: ${e.message}"
                            Toast.makeText(this, "Inference error: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } catch (t: Throwable) {
                Log.e("cpponnxrunner", "bytes infer failed (pre)", t)
                binding.statusMessage.text = "Byte inference failed: ${t.message}"
            }

        } catch (t: Throwable) {
            Log.e("cpponnxrunner", "onActivityResult (bytes path) failed", t)
            binding.statusMessage.text = "Error: ${t.message}"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            releaseSession()
        } catch (t: Throwable) {
            Log.w("cpponnxrunner", "releaseSession failed", t)
        }
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

    // asset -> cache (you may give a different name at the destination)
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

    // Copies a folder (and its subcontents) to cache while preserving the relative structure
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
    // JNI bridges
    // =========================

    external fun createSession(cacheDir: String)
    external fun inferFromBytes(image: ByteArray, mask: ByteArray): ByteArray
    external fun releaseSession()

    companion object {
        init {
            System.loadLibrary("cpponnxrunner")
        }
    }

    external fun cvVersion(): String

    /** copies assets/<assetPath> to $cacheDir/<targetRelative> */
    private fun copyAssetToCache(targetRelative: String, assetPath: String) {
        ensureCacheParents(targetRelative)
        val outFile = File(cacheDir, targetRelative)
        if (outFile.exists()) return
        assets.open(assetPath).use { inp ->
            FileOutputStream(outFile).use { out ->
                inp.copyTo(out)
            }
        }
    }

    /** writes $cacheDir/<relative> (creating parent folders if needed) and returns the absolute path */
    private fun writeBytesToCache(relative: String, data: ByteArray): String {
        ensureCacheParents(relative)
        val outFile = File(cacheDir, relative)
        FileOutputStream(outFile).use { it.write(data) }
        return outFile.absolutePath
    }

    /** creates parent directories for $cacheDir/<relative> */
    private fun ensureCacheParents(relative: String) {
        val parent = File(cacheDir, relative).parentFile
        if (parent != null && !parent.exists()) parent.mkdirs()
    }
}
