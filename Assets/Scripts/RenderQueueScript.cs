using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderQueueScript : MonoBehaviour
{
    public Material shoe_right_material;
    public Material fake_feet_material;
    void Start()
    {
        shoe_right_material.renderQueue = 2000;
        fake_feet_material.renderQueue = 2020;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
