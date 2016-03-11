/*

Copyright (c) 2014, Dr Franck P. Vidal (franck.p.vidal@fpvidal.net),
http://www.fpvidal.net/
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the Bangor University nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


/**
********************************************************************************
*
*	@file		welsh_dragon_glfw.cxx
*
*	@brief		Main test program.
*
*	@version	1.0
*
*	@date		11/11/2013
*
*	@author		Dr Franck P. Vidal
*
*	@section	License
*				BSD 3-Clause License.
*
*				For details on use and redistribution please refer
*				to http://opensource.org/licenses/BSD-3-Clause
*
*	@section	Copyright
*				(c) by Dr Franck P. Vidal (franck.p.vidal@fpvidal.net),
*				http://www.fpvidal.net/, Dec 2014, 2014, version 1.0,
*				BSD 3-Clause License
*
********************************************************************************
*/


//******************************************************************************
//	Include
//******************************************************************************
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#include <GL/glew.h>
#endif

#ifdef __APPLE__
#define GL3_PROTOTYPES 1
#else
#define GL_GLEXT_PROTOTYPES 1
#endif

#define GLFW_INCLUDE_GLCOREARB 1

#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <ctime>

#include "config.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>
#include <fcntl.h>

#ifdef HAS_BUNDLE
#include <boost/filesystem.hpp>
#endif



#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkIntensityWindowingImageFilter.h"




#ifndef GVXR_TYPES_H
#include "gVirtualXRay/Types.h"
#endif

#ifndef GVXR_UNITS_H
#include "gVirtualXRay/Units.h"
#endif

#ifndef GVXR_POLYGON_MESH_H
#include "gVirtualXRay/PolygonMesh.h"
#endif

#ifndef GVXR_CUBE_MESH_H
#include "gVirtualXRay/CubeMesh.h"
#endif

#ifndef GVXR_XRAY_BEAM_H
#include "gVirtualXRay/XRayBeam.h"
#endif

#ifndef GVXR_XRAY_DETECTOR_H
#include "gVirtualXRay/XRayDetector.h"
#endif

#ifndef GVXR_OPENGL_UTILITIES_H
#include "gVirtualXRay/OpenGLUtilities.h"
#endif

#ifndef GVXR_MATRIX4X4_H
#include "gVirtualXRay/Matrix4x4.h"
#endif

#ifndef GVXR_XRAY_RENDERER_H
#include "gVirtualXRay/XRayRenderer.h"
#endif

#ifndef GVXR_SHADER_H
#include "gVirtualXRay/Shader.h"
#endif

#ifndef GVXR_UTILITIES_H
#include "gVirtualXRay/Utilities.h"
#endif

#ifndef GVXR_EXCEPTION_H
#include "gVirtualXRay/Exception.h"
#endif

#ifndef GVXR_STEREO_HELPER_H
#include "gVirtualXRay/StereoHelper.h"
#endif

#ifndef GVXR_TEXT_RENDERER_H
#include "gVirtualXRay/TextRenderer.h"
#endif

#include "display_gl2.frag.h"
#include "display_gl2.vert.h"

#include "display_gl3.frag.h"
#include "display_gl3.vert.h"

#include <boost/filesystem.hpp>
//******************************************************************************
//	Name space
//******************************************************************************
using namespace gVirtualXRay;
namespace fs = boost::filesystem;

//******************************************************************************
//	Defines
//******************************************************************************


//******************************************************************************
//	Constant variables
//******************************************************************************




//******************************************************************************
//	Global variables
//******************************************************************************







bool g_is_xray_image_up_to_date(false);


//******************************************************************************
//	Global init
//******************************************************************************


XRayBeam g_xray_beam;
XRayDetector g_xray_detector;
XRayRenderer g_xray_renderer;

GLFWwindow* g_p_main_window_id(0);

Matrix4x4<GLfloat> g_scene_rotation_matrix;
Matrix4x4<GLfloat> g_detector_rotation_matrix;
Matrix4x4<GLfloat> g_sample_rotation_matrix;

//******************************************************************************
//	Function declaration
//******************************************************************************
void errorCallback(int error, const char* description);
void quit();
void loadDetector(xray::XRayConfig&);
void loadSource(xray::XRayConfig&);
void loadXRaySimulator();
void loadSTLFile(const std::string& aPrefix, PolygonMesh&, double);
void updateXRayImage(const std::string& fname, float);


void initGL(){

	    // Set an error callback
	    glfwSetErrorCallback(errorCallback);

	    // Register the exit callback
		atexit(quit);

		// Initialize GLFW
	    if (!glfwInit())
	    {
	    	throw Exception(__FILE__, __FUNCTION__, __LINE__, "ERROR: cannot initialise GLFW.");
	    }

        // Enable OpenGL 2.1 if possible
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	    // Enable quad-buffering if possible
	    glfwWindowHint(GLFW_STEREO, GL_TRUE);

	    // Enable anti-aliasing
	    glfwWindowHint(GLFW_SAMPLES, 4);

		// Create the window title
		std::stringstream window_title;
		window_title << "gVirtualXRay -- DragonDemo -- GLFW";

		//Hidden window
		glfwWindowHint(GLFW_VISIBLE, GL_FALSE);


		//TODO(Jesse Lovitt): Figure out how to make a non-window GL context to use for rendering an image.
	    // Create a windowed mode window and its OpenGL context
	    g_p_main_window_id = glfwCreateWindow(1, 1, window_title.str().data(), NULL, NULL);

        // Window cannot be created 
        if (!g_p_main_window_id)
	    {
            // Disable quad-buffering
            glfwWindowHint(GLFW_STEREO, GL_FALSE);

            // Create the window
	        g_p_main_window_id = glfwCreateWindow(1, 1, window_title.str().data(), NULL, NULL);
        }
        
        // Window cannot be created 
        if (!g_p_main_window_id)
	    {
	        glfwTerminate();
	    	throw Exception(__FILE__, __FUNCTION__, __LINE__, "ERROR: cannot create a GLFW windowed mode window and its OpenGL context.");
	    }

	    // Make the window's context current
	    glfwMakeContextCurrent(g_p_main_window_id);
}

//-----------------------------
int main(int argc, char** argv)
//-----------------------------
{
    try
    {
    	initGL();

    	if (argc != 2){
    		std::cout<< "Use: expose <path to config.prototxt>\n";
    		exit(1);
    	}
    	xray::Config config;
    	int fin = open(argv[1], O_RDONLY);
    	google::protobuf::io::FileInputStream config_fin(fin);
    	google::protobuf::TextFormat::Parse(&config_fin, &config);
    	xray::XRayConfig xrayconfig = config.xrayconfig();

//    	double dunit = 1.0;
//    	switch (xrayconfig.distance_unit()){
//    	case xray::DistanceUnit::mm:
//    		dunit = mm;
//    		break;
//    	case xray::DistanceUnit::cm:
//    		dunit = cm;
//    		break;
//    	}


		// Load the data
		loadDetector(xrayconfig);
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		loadSource(xrayconfig);
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		loadXRaySimulator();
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		// Add the geometry to the X-ray renderer
		std::vector<PolygonMesh> polymeshes();

		fs::path p(xrayconfig.input_path());
		fs::path output_prefix(xrayconfig.output_path());
	    if(!exists(p) || !is_directory(p)) {
	        std::cout << p << " is not a valid path\n";
	        return 1;
	    }
		fs::directory_iterator begin(p), end;
	    std::vector<fs::directory_entry> v(begin, end);
	    std::sort(v.begin(), v.end());
	    for(std::vector<fs::directory_entry>::iterator it = v.begin(); it != v.end(); ++it){
	    	if ((*it).status().type() == fs::directory_file ){
	    		std::cout << "Processsing: "<< (*it).path().native() << '\n';
	    		fs::directory_iterator stlbegin((*it).path().native()), stlend;
	    	    std::vector<fs::directory_entry> currentvec(stlbegin, stlend);
	    	    for(std::vector<fs::directory_entry>::iterator stl = currentvec.begin(); stl != currentvec.end(); ++stl){
					PolygonMesh* mesh = new PolygonMesh();
	    	    	loadSTLFile((*stl).path().native(), *mesh, xrayconfig.hounsfieldvalue());
					checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
					g_xray_renderer.addInnerSurface(mesh);
	    	    }
				// Update the X-ray image
				fs::path out = output_prefix;
				out /= (*it).path().stem();
				out.replace_extension(".png");
				std::cout <<"Saving to: "<< out << '\n';
				updateXRayImage(out.native(), xrayconfig.detector_saturation_value());
				g_xray_renderer.removeInnerSurfaces();
	    	}
	    }

	}
	// Catch exception if any
	catch (const std::exception& error)
	{
		std::cerr << error.what() << std::endl;
	}

	// Close the window and shut GLFW
	if (g_p_main_window_id)
	{
		glfwDestroyWindow(g_p_main_window_id);
		g_p_main_window_id = 0;
		glfwTerminate();
	}

	// Return an exit code
	return (EXIT_SUCCESS);
}


//---------
void quit()
//---------
{
	if (g_p_main_window_id)
	{
		glfwDestroyWindow(g_p_main_window_id);
        g_p_main_window_id = 0;
        glfwTerminate();
	}
}



//------------------------------------------
void loadSTLFile(const std::string& fname, PolygonMesh& mesh, double hounsfieldval)
//------------------------------------------
{
	std::string stl_filename(fname);

	// Open the file
	FILE* p_file_descriptor(fopen(stl_filename.data(), "rb"));

	// The file is not in the same directory as the executable
	if (!p_file_descriptor)
	{
		// Use the prefix
		std::cout << "!!! "<< fname << "NOT FOUND"<< "\n";
	}
	else
	{
		fclose(p_file_descriptor);
	}

   	// Set geometry
	mesh.setFilename(stl_filename.data());
	mesh.loadSTLFile(false, true, false, true, cm, GL_STATIC_DRAW);
	mesh.mergeVertices(false);
	mesh.setHounsfieldValue(hounsfieldval);
//	mesh.applyScale(3.0);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//-----------------
void loadDetector(xray::XRayConfig& config)
//-----------------
{

	g_xray_detector.setResolutionInUnitsOfLengthPerPixel(cm/(config.detector_resolution()));


	unsigned int pxx = static_cast<unsigned int>(config.detector_resolution()*config.detector_size().x());
	unsigned int pxy = static_cast<unsigned int>(config.detector_resolution()*config.detector_size().y());
	Vec2ui g_number_of_pixels(pxx, pxy);
	g_xray_detector.setNumberOfPixels(g_number_of_pixels);

	VEC3 pos(config.detector_position().x()*cm,config.detector_position().y()*cm,config.detector_position().z()*cm);
	g_xray_detector.setDetectorPosition(pos);

	VEC3 up(config.detector_up().x()*cm,config.detector_up().y()*cm,config.detector_up().z()*cm);
	up.normalize();
	g_xray_detector.setUpVector(up);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//---------------
void loadSource(xray::XRayConfig& config)
//---------------
{
	// Set the energy
	g_xray_beam.initialise(config.energy() *keV);

	// Set the source position
//	g_xray_detector.setXrayPointSource(g_source_position);
	//g_xray_detector.setParallelBeam();
//	g_xray_detector.setPointSource();
	VEC3 pos(config.source_position().x()*cm,config.source_position().y()*cm,config.source_position().z()*cm);
	g_xray_detector.setSquareSource(pos, config.emitter_samples_per_dimension(), config.emitter_size() *cm);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//----------------------
void loadXRaySimulator()
//----------------------
{
	// Initialise the X-ray renderer
    g_xray_renderer.initialise(XRayRenderer::OPENGL,
            GL_RGB16F,
            &g_xray_detector,
            &g_xray_beam);

	g_xray_renderer.useNegativeFilteringFlag(!g_xray_renderer.getNegativeFilteringFlag());

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//--------------------
void writeImage(XRayRenderer::PixelType* im, float detectorsaturation, const char* fname)
//--------------------
{
	typedef itk::Image<XRayRenderer::PixelType, 2> InputImageType;

	typedef  unsigned char OutputPixelType;
	typedef itk::Image<OutputPixelType, 2> OutputImageType;

	//Get buffer into itk Image
	InputImageType::Pointer image = InputImageType::New();
	InputImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	InputImageType::SizeType size;
	size[0] = g_xray_detector.getNumberOfPixels().getX();
	size[1] = g_xray_detector.getNumberOfPixels().getY();
	InputImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions(region);
	image->Allocate();
	itk::ImageRegionIterator<InputImageType> it(image, region);
	it.GoToBegin();
	while( !it.IsAtEnd()){
		it.Set( *im);
		++it;
		++im;
	}

	//Get min and max of scaled
//	float scalefactor = itk::NumericTraits<OutputPixelType>::max()/detectorsaturation;
	typedef itk::MinimumMaximumImageCalculator<InputImageType> CalcType;
	CalcType::Pointer minmaxcalc = CalcType::New();
	minmaxcalc->SetImage(image);
	minmaxcalc->Compute();
	std::cout << "Max flux: " << minmaxcalc->GetMaximum() << ", Min flux: " << minmaxcalc->GetMinimum() << "\n";
//	XRayRenderer::PixelType min = minmaxcalc->GetMinimum()*scalefactor;
//	XRayRenderer::PixelType max = minmaxcalc->GetMaximum()*scalefactor;


//	typedef itk::RescaleIntensityImageFilter< InputImageType, InputImageType> RescaleType;
//	RescaleType::Pointer rescale = RescaleType::New();
//	rescale->SetInput(image);
//	rescale->SetOutputMinimum(min);
//	rescale->SetOutputMaximum(max);

	typedef itk::IntensityWindowingImageFilter< InputImageType, InputImageType> WindowType;
	WindowType::Pointer window = WindowType::New();
	window->SetInput(image);
	window->SetWindowMinimum(0);
	window->SetWindowMaximum(detectorsaturation);
	window->SetOutputMinimum(0);
	window->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());

	typedef itk::CastImageFilter<  InputImageType, OutputImageType > FilterType;
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput( window->GetOutput() );

	typedef itk::ImageFileWriter< OutputImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName( fname );
	writer->SetInput( filter->GetOutput() );

	try
	{
		writer->Update();
	}
	catch( itk::ExceptionObject & e )
	{
		std::cerr << "Error: " << e << std::endl;
	}
}

//--------------------
void updateXRayImage(const std::string& fname, float detectorsaturation)
//--------------------
{
	// The X-ray image is not up-to-date
	if (!g_is_xray_image_up_to_date)
	{
		// Compute the X-Ray image
		g_xray_renderer.computeImage(g_sample_rotation_matrix);

//		std::string ext = ".png";
		// Normalise the X-ray image
		//g_xray_renderer.normalise();

		writeImage( g_xray_renderer.getXRayImage(), detectorsaturation, fname.c_str());
//		g_xray_renderer.printLBuffer(OUTDIR"printLBuffer"+ext);
//		g_xray_renderer.printSumMuxDx(OUTDIR"printSumMuxDx"+ext);
//		g_xray_renderer.printEnergyFluence(OUTDIR"printEnergyFluence"+ext);
//		g_xray_renderer.printXRayImage (OUTDIR"printXRayImage"+ext);

		// The X-ray image is up-to-date
		g_is_xray_image_up_to_date = true;
	}
}

//----------------------------------------------------
void errorCallback(int error, const char* description)
//----------------------------------------------------
{
	std::cerr << "GLFW error: " << description << std::endl;
}


