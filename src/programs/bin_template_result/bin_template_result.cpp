#include "../../core/core_headers.h"

class
        IntegratePeakApp : public MyApp {
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

Peak CreatePeakFromScaledMip(Image& scaled_mip, long pixel_counter, float distance_from_origin, int i, int j, float wanted_min_radius = 0.0, float wanted_max_radius = FLT_MAX) {
    const int wanted_min_distance_from_edges = 0;
    // TODO edge =  extracted box size later?
    const int i_min = wanted_min_distance_from_edges;
    const int i_max = scaled_mip.logical_x_dimension - wanted_min_distance_from_edges;
    const int j_min = wanted_min_distance_from_edges;
    const int j_max = scaled_mip.logical_y_dimension - wanted_min_distance_from_edges;

    Peak current_peak;

    if ( distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius && i >= i_min && i <= i_max && j >= j_min && j <= j_max ) {
        current_peak.value                         = scaled_mip.real_values[pixel_counter];
        current_peak.x                             = i - scaled_mip.physical_address_of_box_center_x;
        current_peak.y                             = j - scaled_mip.physical_address_of_box_center_y;
        current_peak.z                             = 0.0;
        current_peak.physical_address_within_image = pixel_counter;
        //wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z, found_peak.value);
        //wxPrintf("new peak %i, %i, %i (%f)\n", i, j, k, found_peak.value);
    }

    return current_peak;
}

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

IMPLEMENT_APP(IntegratePeakApp)

// override the DoInteractiveUserInput

void IntegratePeakApp::DoInteractiveUserInput( ) {
    wxString scaled_mip_input_filename;
    wxString scaled_mip_output_filename;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;
    ;
    float defocus_angle;
    float phase_shift;
    float low_resolution_limit    = 300.0f;
    float high_resolution_limit   = 8.0f;
    float angular_range           = 2.0f;
    float angular_step            = 5.0f;
    int   best_parameters_to_keep = 20;
    float defocus_search_range    = 1000;
    float defocus_search_step     = 10;
    //	float 		defocus_refine_step = 5;
    float pixel_size_search_range = 0.1f;
    float pixel_size_step         = 0.001f;
    //	float		pixel_size_refine_step = 0.001f;
    float    padding               = 1.0;
    bool     ctf_refinement        = false;
    float    mask_radius           = 0.0f;
    wxString my_symmetry           = "C1";
    float    in_plane_angular_step = 0;
    float    wanted_threshold;
    float    min_peak_radius;
    float    xy_change_threshold        = 10.0f;
    bool     exclude_above_xy_threshold = false;
    int      result_number              = 1;
    int      box_size                   = 32;

    int max_threads;

    UserInput* my_input = new UserInput("IntegratePeak", 1.00);

    scaled_mip_input_filename  = my_input->GetFilenameFromUser("Input scaled MIP file", "The file with the scaled MIP (peak search done on this image)", "scaled_mip.mrc", false);
    scaled_mip_output_filename = my_input->GetFilenameFromUser("Output scaled MIP file", "The file with the scaled MIP (recalculated using binning)", "output_scaled_mip.mrc", false);
    wanted_threshold           = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
    min_peak_radius            = my_input->GetFloatFromUser("Min peak radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);
    // TODO: min peak radius not useful
    pixel_size    = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    padding       = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    mask_radius   = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
    result_number = my_input->GetIntFromUser("Result number to refine", "If input files contain results from several searches, which one should be refined?", "1", 1);
    box_size      = my_input->GetIntFromUser("Box size for peak binning", "todo", "32", 1, 512);

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    wxString filename_for_gui_result_image;

    wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive

    delete my_input;

    //	my_current_job.Reset(42);
    my_current_job.ManualSetArguments("ttffffftiii",
                                      scaled_mip_input_filename.ToUTF8( ).data( ),
                                      scaled_mip_output_filename.ToUTF8( ).data( ),
                                      pixel_size,
                                      padding,
                                      mask_radius,
                                      wanted_threshold,
                                      min_peak_radius,
                                      my_symmetry.ToUTF8( ).data( ),
                                      result_number,
                                      max_threads,
                                      box_size);
}

// override the do calculation method which will be what is actually run..

bool IntegratePeakApp::DoCalculation( ) {
    wxDateTime start_time                 = wxDateTime::Now( );
    wxString   scaled_mip_input_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString   scaled_mip_output_filename = my_current_job.arguments[1].ReturnStringArgument( );
    float      pixel_size                 = my_current_job.arguments[2].ReturnFloatArgument( );
    float      padding                    = my_current_job.arguments[3].ReturnFloatArgument( );
    float      mask_radius                = my_current_job.arguments[4].ReturnFloatArgument( );
    float      wanted_threshold           = my_current_job.arguments[5].ReturnFloatArgument( );
    float      min_peak_radius            = my_current_job.arguments[6].ReturnFloatArgument( );
    wxString   my_symmetry                = my_current_job.arguments[7].ReturnStringArgument( );
    int        result_number              = my_current_job.arguments[8].ReturnIntegerArgument( );
    int        max_threads                = my_current_job.arguments[9].ReturnIntegerArgument( );
    int        box_size                   = my_current_job.arguments[10].ReturnIntegerArgument( );

    if ( is_running_locally == false ) {
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...
    }

    int i, j;

    long  thread_pixel_counter = 0;
    long  pixel_counter        = 0;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  best_address;

    int current_x;
    int current_y;

    AnglesAndShifts          angles;
    TemplateComparisonObject template_object;

    ImageFile scaled_mip_input_file;
    ImageFile input_reconstruction_file;

    scaled_mip_input_file.OpenFile(scaled_mip_input_filename.ToStdString( ), false);

    Image input_image;
    Image windowed_particle;
    Image scaled_mip_for_peak_extraction;
    Image scaled_mip_unmasked;
    Image extracted_scaled_mip_patch;
    Image scaled_mip_recalculated;
    Image best_scaled_mip;

    Image scaled_mip_image;
    Peak  current_peak;
    Peak  template_peak;
    Peak  best_peak;
    long  current_address;

    float best_score;
    float starting_score;
    bool  first_score;

    int   peak_number;
    float mask_falloff     = 20.0;
    float min_peak_radius2 = powf(min_peak_radius, 2);

    int jj, ii;

    scaled_mip_image.ReadSlice(&scaled_mip_input_file, result_number);

    scaled_mip_for_peak_extraction.Allocate(scaled_mip_image.logical_x_dimension, scaled_mip_image.logical_y_dimension, 1);
    scaled_mip_recalculated.Allocate(scaled_mip_image.logical_x_dimension, scaled_mip_image.logical_y_dimension, 1);

    scaled_mip_recalculated.SetToConstant(0.0f);

    // allocate windowed particle with user input box size

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    //Peak* found_peaks = new Peak[scaled_mip_image.logical_x_dimension * scaled_mip_image.logical_y_dimension / 100];

    // if running locally, search over all of them
    best_scaled_mip.CopyFrom(&scaled_mip_image);
    scaled_mip_unmasked.CopyFrom(&scaled_mip_image);
    current_peak.value = FLT_MAX;
    float snr_recalculated;

    //test
    // wxPrintf("real_space pixel: %ld\n", best_scaled_mip.real_memory_allocated); // TODO what is difference between real memory and number of real pixels
    // wxPrintf("number of real space pixels: %ld\n", best_scaled_mip.number_of_real_space_pixels);
    // for ( pixel_counter = 0; pixel_counter < best_scaled_mip.real_memory_allocated; pixel_counter++ ) {
    // TODO: loop through real space pixels, how to generate a peak using it?
    //     if ( pixel_counter >= 5758.0 && pixel_counter <= 5762.0 ) {
    //        wxPrintf("pixel %ld, val = %f\n", pixel_counter, best_scaled_mip.real_values[pixel_counter]);
    //   }
    // }
    // if ( best_scaled_mip.object_is_centred_in_box ) wxPrintf("is centered in box\n");

    // to create a peak from pixel counter omp optimize the outer loop first
    float x, y, distance_from_origin;

    /*
    while ( current_peak.value >= wanted_threshold ) {
        // find peak using masked scaled mip so that peaks don't get picked repeatedly
        current_peak = best_scaled_mip.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction_file.ReturnXSize( ) / cistem::fraction_of_box_size_to_exclude_for_border + 1); // edge is 97
        if ( current_peak.value < wanted_threshold )
            break;
        found_peaks[number_of_peaks_found] = current_peak;

        // allocate a new patch o.w. its size will be /2 each time a peak is selected
        extracted_scaled_mip_patch.Allocate(box_size, box_size, true);

        // clip highest peak patch from scaled mip into another image
        scaled_mip_for_peak_extraction.CopyFrom(&scaled_mip_unmasked); // TODO: is scaled_mip_unmasked necessary?
        scaled_mip_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
        scaled_mip_for_peak_extraction.ClipInto(&extracted_scaled_mip_patch);
        //extracted_scaled_mip_patch.QuickAndDirtyWriteSlice(wxString::Format("peak_%i.mrc", number_of_peaks_found).ToStdString( ), 1);

        // do real space binning on extracted scaled_mip patch
        extracted_scaled_mip_patch.RealSpaceBinning(binning_factor, binning_factor);
        //extracted_scaled_mip_patch.QuickAndDirtyWriteSlice(wxString::Format("peak_%i_after_binning.mrc", number_of_peaks_found).ToStdString( ), 1);

        // calculate the new SNR value for this pixel
        snr_recalculated = extracted_scaled_mip_patch.ReturnAverageOfRealValues( ) * extracted_scaled_mip_patch.number_of_real_space_pixels;

        // mask out best_scaled_mip center pixel (only a single pixel not the entire circle) so it won't be picked again..

        float sq_dist_x, sq_dist_y;
        address = 0;

        current_peak.x = current_peak.x + best_scaled_mip.physical_address_of_box_center_x;
        current_peak.y = current_peak.y + best_scaled_mip.physical_address_of_box_center_y;

        //		wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

        for ( j = 0; j < best_scaled_mip.logical_y_dimension; j++ ) {
            sq_dist_y = float(pow(j - current_peak.y, 2));
            for ( i = 0; i < best_scaled_mip.logical_x_dimension; i++ ) {
                sq_dist_x = float(pow(i - current_peak.x, 2));

                // fill another scaled mip image using the new snr values

                if ( sq_dist_x == 0.0f && sq_dist_y == 0.0f ) {
                    // mask out central pixel so it won't be picked again..
                    best_scaled_mip.real_values[address] = -FLT_MAX;
                    // in the recalculated scaled mip fill in the new snr
                    scaled_mip_recalculated.real_values[address] = snr_recalculated;
                }

                address++;
            }
            address += best_scaled_mip.padding_jump_value;
        }

        number_of_peaks_found++;

        wxPrintf("Peak %4i at x, y =  %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x * pixel_size, current_peak.y * pixel_size, current_peak.value);
    }
    */
    //TODO: optimize workflow using openMP
    // TODO: could the result be modeled using GEV?
    float old_snr_value;
#pragma omp parallel num_threads(max_threads) default(none) shared(scaled_mip_output_filename, box_size, best_scaled_mip, scaled_mip_unmasked, scaled_mip_recalculated, max_threads) private(sq_dist_y, sq_dist_x, scaled_mip_for_peak_extraction, distance_from_origin, i, j, ii, jj, current_peak, extracted_scaled_mip_patch, snr_recalculated, old_snr_value, thread_pixel_counter)
    {

//	while (current_peak.value >= wanted_threshold)
#pragma omp for schedule(dynamic, 1)
        for ( j = 0; j < best_scaled_mip.logical_y_dimension; j++ ) {
            sq_dist_y = powf(j - best_scaled_mip.physical_address_of_box_center_y, 2);

            for ( i = 0; i < best_scaled_mip.logical_x_dimension; i++ ) {
                sq_dist_x = powf(i - best_scaled_mip.physical_address_of_box_center_x, 2);

                distance_from_origin = sq_dist_x + sq_dist_y;

                // recalculate pixel location (redundant but for OMP threads)
                thread_pixel_counter = 0;
                for ( jj = 0; jj < best_scaled_mip.logical_y_dimension; jj++ ) {
                    for ( ii = 0; ii < best_scaled_mip.logical_x_dimension; ii++ ) {
                        if ( jj == j && ii == i ) {
                            goto COUNTEREND;
                        }
                        thread_pixel_counter++;
                    }
                    thread_pixel_counter += best_scaled_mip.padding_jump_value;
                }

            COUNTEREND:
                //  create peak using location
                current_peak = CreatePeakFromScaledMip(best_scaled_mip, thread_pixel_counter, distance_from_origin, i, j);

                // now that we have a peak...
                // extract patch
                extracted_scaled_mip_patch.Allocate(box_size, box_size, true);
                scaled_mip_for_peak_extraction.CopyFrom(&scaled_mip_unmasked); // TODO: is scaled_mip_unmasked necessary?
                scaled_mip_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
                scaled_mip_for_peak_extraction.ClipInto(&extracted_scaled_mip_patch);
                // wxPrintf("snr before binning %f\n", extracted_scaled_mip_patch.ReturnCentralPixelValue( ));
                // do real space binning on extracted scaled_mip patch
                // extracted_scaled_mip_patch.RealSpaceBinning(binning_factor, binning_factor);

                // calculate the new SNR value for this pixel
                snr_recalculated = extracted_scaled_mip_patch.ReturnSumOfRealValues( );
                // wxPrintf("snr after binning %f\n", snr_recalculated);

                // mask out pixel in best_scaled_mip
                // TODO: criticle or together
                //old_snr_value                                     = best_scaled_mip.real_values[thread_pixel_counter];
                best_scaled_mip.real_values[thread_pixel_counter] = -FLT_MAX;

                // fill in new scaled mip with recalculated SNR values
                // TODO: criticle or together
                scaled_mip_recalculated.real_values[thread_pixel_counter] = snr_recalculated;
                //wxPrintf("old/new snr value: %f/%f\n", old_snr_value, snr_recalculated);
            }
            //#pragma omp critical
        }

    } // end omp section

    scaled_mip_recalculated.QuickAndDirtyWriteSlice(scaled_mip_output_filename.ToStdString( ), 1);
    //	delete [] addresses;

    if ( is_running_locally == true ) {
        wxPrintf("\nRefine Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
