package com.example.cpponnxrunner

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.util.Log

object GlInfo {
    fun query(): Pair<String?, String?> {
        // 1) Display
        val display: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val version = IntArray(2)
        if (!EGL14.eglInitialize(display, version, 0, version, 1)) {
            Log.w("GLINFO", "eglInitialize failed")
            return null to null
        }

        // 2) Config (ES2)
        val attribs = intArrayOf(
            EGL14.EGL_RED_SIZE,   8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE,  8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numCfg = IntArray(1)
        EGL14.eglChooseConfig(display, attribs, 0, configs, 0, 1, numCfg, 0)

        // 3) Pbuffer surface (küçük, görünmez)
        val pbufAttribs = intArrayOf(
            EGL14.EGL_WIDTH,  1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
        val surface: EGLSurface = EGL14.eglCreatePbufferSurface(display, configs[0], pbufAttribs, 0)

        // 4) Context (ES2)
        val ctxAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )
        val ctx: EGLContext = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)

        // 5) Make current -> glGetString çağrıları ARTIK geçerli
        EGL14.eglMakeCurrent(display, surface, surface, ctx)

        val vendor   = GLES20.glGetString(GLES20.GL_VENDOR)
        val renderer = GLES20.glGetString(GLES20.GL_RENDERER)
        val versionStr = GLES20.glGetString(GLES20.GL_VERSION)
        Log.i("GLINFO", "VENDOR=$vendor | RENDERER=$renderer | VERSION=$versionStr")

        // 6) Temizlik
        EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(display, surface)
        EGL14.eglDestroyContext(display, ctx)
        EGL14.eglTerminate(display)

        return vendor to renderer
    }
}
