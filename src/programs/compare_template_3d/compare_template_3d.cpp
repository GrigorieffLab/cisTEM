#include "../../core/core_headers.h"

class
        CompareTemplate3DApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

class TemplateComparisonObject {
  public:
    Image *          input_reconstruction, *windowed_particle, *projection_filter;
    AnglesAndShifts* angles;
    float            pixel_size_factor;
    //	int							slice = 1;
};

// This is the function which will be minimized
Peak TemplateScore(void* scoring_parameters) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    Image                     current_projection;
    //	Peak box_peak;

    current_projection.Allocate(comparison_object->projection_filter->logical_x_dimension, comparison_object->projection_filter->logical_x_dimension, false);
    if ( comparison_object->input_reconstruction->logical_x_dimension != current_projection.logical_x_dimension ) {
        Image padded_projection;
        padded_projection.Allocate(comparison_object->input_reconstruction->logical_x_dimension, comparison_object->input_reconstruction->logical_x_dimension, false);
        comparison_object->input_reconstruction->ExtractSlice(padded_projection, *comparison_object->angles, 1.0f, false);
        padded_projection.SwapRealSpaceQuadrants( );
        padded_projection.BackwardFFT( );
        padded_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
        //		padded_projection.ChangePixelSize(&padded_projection, comparison_object->pixel_size_factor, 0.001f);
        //		padded_projection.ClipInto(&current_projection);
        //		current_projection.ForwardFFT();
    }
    else {
        comparison_object->input_reconstruction->ExtractSlice(current_projection, *comparison_object->angles, 1.0f, false);
        current_projection.SwapRealSpaceQuadrants( );
        current_projection.BackwardFFT( );
        current_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
    }

    //	current_projection.QuickAndDirtyWriteSlice("projections.mrc", comparison_object->slice);
    //	comparison_object->slice++;
    current_projection.MultiplyPixelWise(*comparison_object->projection_filter);
    //	current_projection.BackwardFFT();
    //	current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());
    //	current_projection.Resize(comparison_object->windowed_particle->logical_x_dimension, comparison_object->windowed_particle->logical_y_dimension, 1, 0.0f);
    //	current_projection.ForwardFFT();
    current_projection.ZeroCentralPixel( );
    current_projection.DivideByConstant(sqrtf(current_projection.ReturnSumOfSquares( )));
#ifdef MKL
    // Use the MKL
    vmcMulByConj(current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        current_projection.complex_values[pixel_counter] = std::conj(current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }
#endif
    current_projection.BackwardFFT( );
    //	wxPrintf("ping");

    return current_projection.FindPeakWithIntegerCoordinates( );
    //	box_peak = current_projection.FindPeakWithIntegerCoordinates();
    //	wxPrintf("address = %li\n", box_peak.physical_address_within_image);
    //	box_peak.x = 0.0f;
    //	box_peak.y = 0.0f;
    //	box_peak.value = current_projection.real_values[33152];
    //	return box_peak;
}

IMPLEMENT_APP(CompareTemplate3DApp)

// override the DoInteractiveUserInput

void CompareTemplate3DApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction_1, input_reconstruction_2;
    wxString data_directory_name;
    float    high_resolution_limit;

    UserInput* my_input = new UserInput("CompareTemplate3D", 1.00);

    input_search_images    = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction_1 = my_input->GetFilenameFromUser("Input template reconstruction 1", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_2 = my_input->GetFilenameFromUser("Input template reconstruction 2", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    data_directory_name    = my_input->GetFilenameFromUser("Name for data directory", "path to data directory", "60_120_5_2.5", false);
    high_resolution_limit  = my_input->GetFloatFromUser("High resolution limit (0-1)", "corner should be sqrt(2)/2=0.707107", "0.5", 0.0, sqrt(2) / 2);

    delete my_input;

    //	my_current_job.Reset(42);
    my_current_job.ManualSetArguments("ttttf", input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction_1.ToUTF8( ).data( ),
                                      input_reconstruction_2.ToUTF8( ).data( ),
                                      data_directory_name.ToUTF8( ).data( ),
                                      high_resolution_limit);
}

// override the do calculation method which will be what is actually run..

bool CompareTemplate3DApp::DoCalculation( ) {
    wxDateTime start_time = wxDateTime::Now( );

    wxString input_search_images_filename    = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename_1 = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_reconstruction_filename_2 = my_current_job.arguments[2].ReturnStringArgument( );
    wxString data_directory_name             = my_current_job.arguments[3].ReturnStringArgument( );
    float    high_resolution_limit           = my_current_job.arguments[4].ReturnFloatArgument( );

    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file_1;
    ImageFile input_reconstruction_file_2;

    Curve whitening_filter;
    Curve number_of_terms;
    Curve ps_1, ps_2, n1, n2;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_reconstruction_file_1.OpenFile(input_reconstruction_filename_1.ToStdString( ), false);
    input_reconstruction_file_2.OpenFile(input_reconstruction_filename_2.ToStdString( ), false);

    Image input_image;
    Image input_reconstruction_1;
    Image input_reconstruction_2;
    //	Image template_reconstruction;
    //	Image current_projection;
    //	Image padded_projection;

    Image projection_filter;

    input_image.ReadSlice(&input_search_image_file, 1);

    float padding = 1.0;
    input_reconstruction_1.ReadSlices(&input_reconstruction_file_1, 1, input_reconstruction_file_1.ReturnNumberOfSlices( ));
    input_reconstruction_2.ReadSlices(&input_reconstruction_file_2, 1, input_reconstruction_file_2.ReturnNumberOfSlices( ));

    if ( padding != 1.0f ) {
        input_reconstruction_1.Resize(input_reconstruction_1.logical_x_dimension * padding, input_reconstruction_1.logical_y_dimension * padding, input_reconstruction_1.logical_z_dimension * padding, input_reconstruction_1.ReturnAverageOfRealValuesOnEdges( ));
        input_reconstruction_2.Resize(input_reconstruction_2.logical_x_dimension * padding, input_reconstruction_2.logical_y_dimension * padding, input_reconstruction_2.logical_z_dimension * padding, input_reconstruction_2.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction_1.ForwardFFT( );
    input_reconstruction_2.ForwardFFT( );
    //input_reconstruction.CosineMask(0.1, 0.01, true);
    //input_reconstruction.Whiten();
    //if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
    input_reconstruction_1.ZeroCentralPixel( );
    //input_reconstruction_1.SwapRealSpaceQuadrants( );
    input_reconstruction_2.ZeroCentralPixel( );
    input_reconstruction_2.SwapRealSpaceQuadrants( );

    // for now, I am assuming the MTF has been applied already.
    // work out the filter to just whiten the image..

    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    ps_1.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction_1.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    ps_2.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction_2.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    n1.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction_1.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    n2.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction_2.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    // remove outliers

    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));
    //whitening_filter.WriteToFile("filter.txt");

    input_reconstruction_1.GaussianLowPassFilter(10.0);
    input_reconstruction_1.ZeroCentralPixel( );
    input_reconstruction_1.DivideByConstant(sqrtf(input_reconstruction_1.ReturnSumOfSquares( )));

    input_reconstruction_2.GaussianLowPassFilter(10.0);
    input_reconstruction_2.ZeroCentralPixel( );
    input_reconstruction_2.DivideByConstant(sqrtf(input_reconstruction_2.ReturnSumOfSquares( )));

#ifdef MKL
    // Use the MKL
    vmcMulByConj(input_reconstruction_1.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(input_reconstruction_2.complex_values), reinterpret_cast<MKL_Complex8*>(input_reconstruction_1.complex_values), reinterpret_cast<MKL_Complex8*>(input_reconstruction_1.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( int pixel_counter = 0; pixel_counter < input_reconstruction_1.real_memory_allocated / 2; pixel_counter++ ) {
        input_reconstruction_1.complex_values[pixel_counter] = conj(input_reconstruction_1.complex_values[pixel_counter]) * input_reconstruction_2.complex_values[pixel_counter];
    }
#endif

    input_reconstruction_1.SwapRealSpaceQuadrants( );
    input_reconstruction_1.QuickAndDirtyWriteSlices(data_directory_name.ToStdString( ), 1, input_reconstruction_1.logical_z_dimension, true);
    return true;
}