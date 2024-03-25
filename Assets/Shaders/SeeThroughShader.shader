Shader "Custom/SeeThroughShader"

{
    SubShader{
            //Tags {"Queue" = "Geometry-10" }
            Tags {"Queue" = "Geometry-20" }
            Lighting Off
            ZTest LEqual
            ZWrite On
            ColorMask 0
            Pass {}
    }
        Fallback "Diffuse"
}

/*
{
    SubShader
    {
        Tags { "Queue" = "Transparent + 1990" }
        Blend SrcAlpha OneMinusSrcAlpha  // Enable alpha blending
        Lighting Off
        ZTest LEqual
        ZWrite On
        ColorMask 0
        Pass {}
    }
    Fallback "Transparent/Diffuse"  // Fallback to a transparent shader
}
*/