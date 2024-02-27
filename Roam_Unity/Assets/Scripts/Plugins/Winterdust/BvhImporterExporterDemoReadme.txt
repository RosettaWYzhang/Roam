BvhImporterExporterDemo 1.1.0 (Winterdust, Sweden)

Extract the ZIP archive in the "Assets" folder inside your Unity project.
This path should exist after you've extracted: Assets/Plugins/Winterdust

Afterwards you can put this at the top of your scripts to access the BVH class:

	using Winterdust;

Getting started is simple. Here's an example line of code to try on your favorite .bvh file:

	new BVH("C:\\YourFileHere.bvh", -10).makeDebugSkeleton(true, "00ff00");

What it does is import the .bvh file and make a green animated stick figure with a mesh at each joint, moving at 10 FPS.
The path can be absolute or relative to the working directory of the program.
Instead of a path to a .bvh file there's also a constructor accepting the content of a .bvh file directly.

Inside the Unity Editor the working directory is the folder that contains the "Assets" and "ProjectSettings" folders.
If you run a stand-alone build from inside the editor the working directory is the same.
If you run the stand-alone build from outside the editor the working directory will typically be the folder of the executable.
Take this into account if you use relative paths!

When benchmarking remember that the performance of the DLL will increase a lot when you run from a stand-alone build.
The BVH constructor will usually finish around twice as fast outside the Unity Editor.

The demo version has the following limitations:

+ You have to put an empty text file into Assets/Resources by a specific name.
  The first time you make an instance of the BVH class it will print the required name in the console.
  This name needs to be changed every once in a while. You'll be informed by the BVH's constructor when the time comes.
+ The writeToDisk() method is not working in the demo.

Thanks for checking out BvhImporterExporterDemo! Get the full version today!

https://winterdust.itch.io/bvhimporterexporter