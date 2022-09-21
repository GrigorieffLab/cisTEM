#include "../../core/core_headers.h"
#include <iostream>
#include <iomanip>

using namespace Eigen;

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
    wxString best_psi_input_filename;
    wxString best_theta_input_filename;
    wxString best_phi_input_filename;
    wxString input_reconstruction_filename;

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

    scaled_mip_input_filename     = my_input->GetFilenameFromUser("Input scaled MIP file", "The file with the scaled MIP (peak search done on this image)", "scaled_mip.mrc", true);
    best_psi_input_filename       = my_input->GetFilenameFromUser("Input psi file", "The file with the best psi image", "psi.mrc", true);
    best_theta_input_filename     = my_input->GetFilenameFromUser("Input theta file", "The file with the best psi image", "theta.mrc", true);
    best_phi_input_filename       = my_input->GetFilenameFromUser("Input phi file", "The file with the best psi image", "phi.mrc", true);
    input_reconstruction_filename = my_input->GetFilenameFromUser("Template file", "XXX", "output_scaled_mip.mrc", true);
    wanted_threshold              = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
    min_peak_radius               = my_input->GetFloatFromUser("Min peak radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);
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
    my_current_job.ManualSetArguments("tttttffffftiii",
                                      scaled_mip_input_filename.ToUTF8( ).data( ),
                                      best_phi_input_filename.ToUTF8( ).data( ),
                                      best_theta_input_filename.ToUTF8( ).data( ),
                                      best_psi_input_filename.ToUTF8( ).data( ),
                                      input_reconstruction_filename.ToUTF8( ).data( ),
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
    wxDateTime start_time                    = wxDateTime::Now( );
    wxString   scaled_mip_input_filename     = my_current_job.arguments[0].ReturnStringArgument( );
    wxString   best_phi_input_filename       = my_current_job.arguments[1].ReturnStringArgument( );
    wxString   best_theta_input_filename     = my_current_job.arguments[2].ReturnStringArgument( );
    wxString   best_psi_input_filename       = my_current_job.arguments[3].ReturnStringArgument( );
    wxString   input_reconstruction_filename = my_current_job.arguments[4].ReturnStringArgument( );
    float      pixel_size                    = my_current_job.arguments[5].ReturnFloatArgument( );
    float      padding                       = my_current_job.arguments[6].ReturnFloatArgument( );
    float      mask_radius                   = my_current_job.arguments[7].ReturnFloatArgument( );
    float      wanted_threshold              = my_current_job.arguments[8].ReturnFloatArgument( );
    float      min_peak_radius               = my_current_job.arguments[9].ReturnFloatArgument( );
    wxString   my_symmetry                   = my_current_job.arguments[10].ReturnStringArgument( );
    int        result_number                 = my_current_job.arguments[11].ReturnIntegerArgument( );
    int        max_threads                   = my_current_job.arguments[12].ReturnIntegerArgument( );
    int        box_size                      = my_current_job.arguments[13].ReturnIntegerArgument( );
    /*
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    wxPrintf("m=%lf\n", m(0, 0));
    // std::cout << "debug" << std::endl; TODO: ask why std not printing
    */

    //float              rot = deg_2_rad(130.0f), tilt = deg_2_rad(30.0f), psi = deg_2_rad(199.5f);
    //Eigen::Quaternionf q1;
    //Eigen::Quaternionf q2;
    //q1 = Eigen::AngleAxisf(rot, Eigen::Vector3f::UnitZ( )) * Eigen::AngleAxisf(tilt, Eigen::Vector3f::UnitY( )) * Eigen::AngleAxisf(psi, Eigen::Vector3f::UnitZ( ));
    //float distance = acosf(q1.coeffs( )[0] * q2.coeffs( )[0] + q1.coeffs( )[1] * q2.coeffs( )[1] + q1.coeffs( )[2] * q2.coeffs( )[2] + q1.coeffs( )[3] * q2.coeffs( )[3]);
    //wxPrintf("distance = %f degrees\n", rad_2_deg(distance));
    // cisTEM is using ZYZ https://www.ccpem.ac.uk/user_help/rotation_conventions.php

    // now that I can convert between euler and quaternion

    // calculate the "variance" of rotational/angular distribution (is match_template uniformaly sampling or not?) by calculating the distance between rotations of pixel pairs: dSO(3)(q1,q2)=2*cos^(-1)(|q1*q2|) see paper: XXX TODO XXX

    if ( is_running_locally == false ) {
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...
    }

    int i, j;

    long  thread_pixel_counter = 0;
    int   pixel_counter_1      = 0;
    int   pixel_counter_2      = 0;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  best_address;

    int current_x;
    int current_y;

    TemplateComparisonObject template_object;

    ImageFile scaled_mip_input_file;
    ImageFile input_reconstruction_file;
    ImageFile best_psi_input_file;
    ImageFile best_theta_input_file;
    ImageFile best_phi_input_file;

    scaled_mip_input_file.OpenFile(scaled_mip_input_filename.ToStdString( ), false);
    best_psi_input_file.OpenFile(best_psi_input_filename.ToStdString( ), false);
    best_theta_input_file.OpenFile(best_theta_input_filename.ToStdString( ), false);
    best_phi_input_file.OpenFile(best_phi_input_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);
    Image input_image;
    Image scaled_mip_for_peak_extraction;
    Image psi_for_peak_extraction;
    Image theta_for_peak_extraction;
    Image phi_for_peak_extraction;
    Image scaled_mip_unmasked;
    Image extracted_scaled_mip_patch;
    Image extracted_psi_patch;
    Image extracted_theta_patch;
    Image extracted_phi_patch;
    Image best_scaled_mip;

    Image scaled_mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Peak  current_peak;
    Peak  template_peak;
    Peak  best_peak;
    long  current_address;

    float best_score;
    float starting_score;
    bool  first_score;

    int   number_of_peaks_found = 0;
    int   peak_number;
    float mask_falloff     = 20.0;
    float min_peak_radius2 = powf(min_peak_radius, 2);
    int   jj, ii;
    scaled_mip_image.ReadSlice(&scaled_mip_input_file, result_number);
    psi_image.ReadSlice(&best_psi_input_file, result_number);
    theta_image.ReadSlice(&best_theta_input_file, result_number);
    phi_image.ReadSlice(&best_phi_input_file, result_number);

    /*
    float              pixel_1_phi, pixel_1_theta, pixel_1_psi, pixel_2_phi, pixel_2_theta, pixel_2_psi;
    Eigen::Quaternionf pixel_1_q, pixel_2_q;
    float              pairwise_distance;
    NumericTextFile    similarity_file("similarity_peak.txt", OPEN_TO_WRITE, 1);

    for ( pixel_counter_1 = 0; pixel_counter_1 < psi_image.number_of_real_space_pixels; pixel_counter_1++ ) {
        for ( pixel_counter_2 = pixel_counter_1 + 1; pixel_counter_2 < psi_image.number_of_real_space_pixels; pixel_counter_2++ ) {
            pixel_1_phi   = phi_image.real_values[pixel_counter_1];
            pixel_1_theta = theta_image.real_values[pixel_counter_1];
            pixel_1_psi   = psi_image.real_values[pixel_counter_1];
            pixel_2_phi   = phi_image.real_values[pixel_counter_2];
            pixel_2_theta = theta_image.real_values[pixel_counter_2];
            pixel_2_psi   = psi_image.real_values[pixel_counter_2];
            pixel_1_q     = Eigen::AngleAxisf(pixel_1_phi, Eigen::Vector3f::UnitZ( )) * Eigen::AngleAxisf(pixel_1_theta, Eigen::Vector3f::UnitY( )) * Eigen::AngleAxisf(pixel_1_psi, Eigen::Vector3f::UnitZ( ));
            pixel_2_q     = Eigen::AngleAxisf(pixel_2_phi, Eigen::Vector3f::UnitZ( )) * Eigen::AngleAxisf(pixel_2_theta, Eigen::Vector3f::UnitY( )) * Eigen::AngleAxisf(pixel_2_psi, Eigen::Vector3f::UnitZ( ));
            // calculate the "variance" of rotational/angular distribution (is match_template uniformaly sampling or not?) by calculating the distance between rotations of pixel pairs: dSO(3)(q1,q2)=2*cos^(-1)(|q1*q2|) note that I am neglecting factor 2 here. see paper: XXX TODO XXX

            pairwise_distance = pixel_1_q.coeffs( )[0] * pixel_2_q.coeffs( )[0] + pixel_1_q.coeffs( )[1] * pixel_2_q.coeffs( )[1] + pixel_1_q.coeffs( )[2] * pixel_2_q.coeffs( )[2] + pixel_1_q.coeffs( )[3] * pixel_2_q.coeffs( )[3];
            //wxPrintf("%f %f %f %f vs. %f %f %f %f= %f\n", q1.coeffs( )[0], q1.coeffs( )[1], q1.coeffs( )[2], q1.coeffs( )[3], q2.coeffs( )[0], q2.coeffs( )[1], q2.coeffs( )[2], q2.coeffs( )[3], distance);

            if ( fabs(pairwise_distance - 1.0) < 10E-6 ) {
                pairwise_distance = 1.0;
            }
            similarity_file.WriteCommentLine("%i %f %f %f %i %f %f %f %f\n", pixel_counter_1, pixel_1_phi, pixel_1_theta, pixel_1_psi, pixel_counter_2, pixel_2_phi, pixel_2_theta, pixel_2_psi, rad_2_deg(acosf(pairwise_distance)));
        }
    }
    similarity_file.Close( );
    exit(0);
*/
    scaled_mip_for_peak_extraction.Allocate(scaled_mip_image.logical_x_dimension, scaled_mip_image.logical_y_dimension, 1);
    psi_for_peak_extraction.Allocate(psi_image.logical_x_dimension, psi_image.logical_y_dimension, 1);
    theta_for_peak_extraction.Allocate(theta_image.logical_x_dimension, theta_image.logical_y_dimension, 1);
    phi_for_peak_extraction.Allocate(phi_image.logical_x_dimension, phi_image.logical_y_dimension, 1);

    // allocate windowed particle with user input box size

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    Peak* found_peaks = new Peak[scaled_mip_image.logical_x_dimension * scaled_mip_image.logical_y_dimension / 100];

    // if running locally, search over all of them
    best_scaled_mip.CopyFrom(&scaled_mip_image);
    scaled_mip_unmasked.CopyFrom(&scaled_mip_image);
    current_peak.value = FLT_MAX;
    float snr_recalculated;

    float              old_snr_value, phi_1, theta_1, psi_1, phi_2, theta_2, psi_2;
    Eigen::Quaternionf q1, q2;

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

    float x, y, distance_from_origin;

    // store all peaks into found_peaks top 10000 for example (can be user defined)
    while ( current_peak.value >= wanted_threshold ) {
        // find peak using masked scaled mip so that peaks don't get picked repeatedly
        // look for a peak..

        current_peak = best_scaled_mip.FindPeakWithIntegerCoordinates(0.0, FLT_MAX,
                                                                      input_reconstruction_file.ReturnXSize( ) / cistem::fraction_of_box_size_to_exclude_for_border + 1);
        if ( current_peak.value < wanted_threshold )
            break;
        found_peaks[number_of_peaks_found] = current_peak;

        // ok we have peak..

        // get angles and mask out the local area so it won't be picked again..

        float sq_dist_x, sq_dist_y;
        address = 0;

        current_peak.x = current_peak.x + best_scaled_mip.physical_address_of_box_center_x;
        current_peak.y = current_peak.y + best_scaled_mip.physical_address_of_box_center_y;

        //		wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

        for ( j = 0; j < best_scaled_mip.logical_y_dimension; j++ ) {
            sq_dist_y = float(pow(j - current_peak.y, 2));
            for ( i = 0; i < best_scaled_mip.logical_x_dimension; i++ ) {
                sq_dist_x = float(pow(i - current_peak.x, 2));

                // The square centered at the pixel
                if ( sq_dist_x + sq_dist_y <= min_peak_radius2 ) {
                    best_scaled_mip.real_values[address] = -FLT_MAX;
                }

                address++;
            }
            address += best_scaled_mip.padding_jump_value;
        }

        number_of_peaks_found++;

        wxPrintf("Peak %4i at x, y =  %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x * pixel_size, current_peak.y * pixel_size, current_peak.value);
    }

    // distance array
    int length_of_distance_array = 0;
    for ( pixel_counter_1 = 0; pixel_counter_1 < box_size * box_size; pixel_counter_1++ ) {
        for ( pixel_counter_2 = pixel_counter_1 + 1; pixel_counter_2 < box_size * box_size; pixel_counter_2++ ) {
            length_of_distance_array++;
        }
    }
    wxPrintf("length_of_distance_array = %i\n", length_of_distance_array);
    float sum_distance, sum_of_squares_distance, variance, distance;

    //TODO: optimize workflow using openMP
    // TODO: could the result be modeled using GEV?

#pragma omp parallel num_threads(max_threads) default(none) shared(pixel_size, length_of_distance_array, number_of_peaks_found, found_peaks, box_size, scaled_mip_unmasked, psi_image, theta_image, phi_image, max_threads) private(peak_number, scaled_mip_for_peak_extraction, psi_for_peak_extraction, theta_for_peak_extraction, phi_for_peak_extraction, current_peak, extracted_scaled_mip_patch, extracted_psi_patch, extracted_theta_patch, extracted_phi_patch, snr_recalculated, q1, q2, pixel_counter_1, pixel_counter_2, phi_1, theta_1, psi_1, phi_2, theta_2, psi_2, sum_distance, sum_of_squares_distance, variance, distance)
    {
        current_peak.value = FLT_MAX;
        extracted_scaled_mip_patch.Allocate(box_size, box_size, true);
        extracted_psi_patch.Allocate(box_size, box_size, true);
        extracted_theta_patch.Allocate(box_size, box_size, true);
        extracted_phi_patch.Allocate(box_size, box_size, true);

#pragma omp for schedule(dynamic, 1)
        for ( peak_number = 0; peak_number < number_of_peaks_found; peak_number++ ) {
            current_peak = found_peaks[peak_number];
            // ok we have peak..
            // TODO: use peak info to extract patch from scaled mip and angle maps
            scaled_mip_for_peak_extraction.CopyFrom(&scaled_mip_unmasked);
            scaled_mip_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
            scaled_mip_for_peak_extraction.ClipInto(&extracted_scaled_mip_patch);

            psi_for_peak_extraction.CopyFrom(&psi_image);
            psi_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
            psi_for_peak_extraction.ClipInto(&extracted_psi_patch);

            theta_for_peak_extraction.CopyFrom(&theta_image);
            theta_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
            theta_for_peak_extraction.ClipInto(&extracted_theta_patch);

            phi_for_peak_extraction.CopyFrom(&phi_image);
            phi_for_peak_extraction.RealSpaceIntegerShift(current_peak.x, current_peak.y);
            phi_for_peak_extraction.ClipInto(&extracted_phi_patch);

            snr_recalculated = extracted_scaled_mip_patch.ReturnSumOfRealValues( );

            sum_distance            = 0.0;
            sum_of_squares_distance = 0.0;

            // calculate variance of distance between pairwise rotations
            for ( pixel_counter_1 = 0; pixel_counter_1 < extracted_phi_patch.number_of_real_space_pixels; pixel_counter_1++ ) {
                for ( pixel_counter_2 = pixel_counter_1 + 1; pixel_counter_2 < extracted_phi_patch.number_of_real_space_pixels; pixel_counter_2++ ) {
                    phi_1   = extracted_phi_patch.real_values[pixel_counter_1];
                    theta_1 = extracted_theta_patch.real_values[pixel_counter_1];
                    psi_1   = extracted_psi_patch.real_values[pixel_counter_1];
                    phi_2   = extracted_phi_patch.real_values[pixel_counter_2];
                    theta_2 = extracted_theta_patch.real_values[pixel_counter_2];
                    psi_2   = extracted_psi_patch.real_values[pixel_counter_2];
                    q1      = Eigen::AngleAxisf(deg_2_rad(phi_1), Eigen::Vector3f::UnitZ( )) * Eigen::AngleAxisf(deg_2_rad(theta_1), Eigen::Vector3f::UnitY( )) * Eigen::AngleAxisf(deg_2_rad(psi_1), Eigen::Vector3f::UnitZ( ));
                    q2      = Eigen::AngleAxisf(deg_2_rad(phi_2), Eigen::Vector3f::UnitZ( )) * Eigen::AngleAxisf(deg_2_rad(theta_2), Eigen::Vector3f::UnitY( )) * Eigen::AngleAxisf(deg_2_rad(psi_2), Eigen::Vector3f::UnitZ( ));
                    // calculate the "variance" of rotational/angular distribution (is match_template uniformaly sampling or not?) by calculating the distance between rotations of pixel pairs: dSO(3)(q1,q2)=2*cos^(-1)(|q1*q2|) note that I am neglecting factor 2 here. see paper: XXX TODO XXX

                    distance = q1.coeffs( )[0] * q2.coeffs( )[0] + q1.coeffs( )[1] * q2.coeffs( )[1] + q1.coeffs( )[2] * q2.coeffs( )[2] + q1.coeffs( )[3] * q2.coeffs( )[3];
                    //wxPrintf("%f %f %f vs. %f %f %f\n", phi_1, theta_1, psi_1, phi_2, theta_2, psi_2);
                    //wxPrintf("%f %f %f %f vs. %f %f %f %f= %f\n", q1.coeffs( )[0], q1.coeffs( )[1], q1.coeffs( )[2], q1.coeffs( )[3], q2.coeffs( )[0], q2.coeffs( )[1], q2.coeffs( )[2], q2.coeffs( )[3], distance);

                    if ( fabs(distance - 1.0) < 10E-6 ) {
                        distance = 1.0;
                    }
                    sum_distance += rad_2_deg(acosf(distance));
                    sum_of_squares_distance += powf(distance, 2);
                }
            }
            //variance = sum_of_squares_distance / length_of_distance_array - pow(sum_distance / length_of_distance_array, 2);
            sum_distance /= length_of_distance_array;
            wxPrintf("Peak %4i at x, y, snr_sum, mean_acos_deg, snr_sum/mean_acos_deg =  %12.6f, %12.6f, %10.6f, %10.6f, %10.6lf, %10.6f\n", peak_number, (current_peak.x + scaled_mip_unmasked.physical_address_of_box_center_x) * pixel_size, (current_peak.y + scaled_mip_unmasked.physical_address_of_box_center_y) * pixel_size, current_peak.value, snr_recalculated, sum_distance, snr_recalculated / sum_distance);
        }

    } // end omp section

    delete[] found_peaks;

    if ( is_running_locally == true ) {
        wxPrintf("\nRefine Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
