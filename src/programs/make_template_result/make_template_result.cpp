#include "../../core/core_headers.h"

#include "../../core/cistem_constants.h"

class
        MakeTemplateResult : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeTemplateResult)

// override the DoInteractiveUserInput

void MakeTemplateResult::DoInteractiveUserInput( ) {

    wxString input_reconstruction_filename;
    wxString input_mip_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString input_best_pixel_size_filename;
    wxString output_result_image_filename;
    wxString output_slab_filename;
    wxString xyz_coords_filename;

    float wanted_threshold;
    float min_peak_radius;
    float slab_thickness;
    float pixel_size;
    float binning_factor;
    int   result_number;
    int   mip_x_dimension = 0;
    int   mip_y_dimension = 0;
    bool  read_coordinates;
    int   ignore_N_pixels_from_the_border = -1;

    UserInput* my_input = new UserInput("MakeTemplateResult", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from file?", "Should the target coordinates be read from a file instead of search results?", "No");
    if ( ! read_coordinates ) {
        input_mip_filename  = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
        xyz_coords_filename = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        wanted_threshold    = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        min_peak_radius     = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        result_number       = my_input->GetIntFromUser("Result number to process", "If input files contain results from several searches, which one should be used?", "1", 1);
    }
    else {
        mip_x_dimension     = my_input->GetIntFromUser("X-dimension of original MIP", "The x-dimension of the MIP that contained the peaks listed in the input coordinate file", "5760", 100);
        mip_y_dimension     = my_input->GetIntFromUser("Y-dimension of original MIP", "The y-dimension of the MIP that contained the peaks listed in the input coordinate file", "4092", 100);
        xyz_coords_filename = my_input->GetFilenameFromUser("Input x,y,z coordinate file", "The file containing the x,y,z coordinates of the found targets", "coordinates.txt", false);
    }
    input_reconstruction_filename   = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    pixel_size                      = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    ignore_N_pixels_from_the_border = my_input->GetIntFromUser("Ignore N pixels from the edge of the MIP", "Defaults to 1/2 the template dimension (-1)", "-1", -1);

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tttfffbiiii", input_reconstruction_filename.ToUTF8( ).data( ),
                                      input_mip_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      wanted_threshold,
                                      min_peak_radius,
                                      pixel_size,
                                      read_coordinates,
                                      mip_x_dimension,
                                      mip_y_dimension,
                                      result_number,
                                      ignore_N_pixels_from_the_border);
}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResult::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_reconstruction_filename   = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_mip_filename              = my_current_job.arguments[1].ReturnStringArgument( );
    wxString xyz_coords_filename             = my_current_job.arguments[2].ReturnStringArgument( );
    float    wanted_threshold                = my_current_job.arguments[3].ReturnFloatArgument( );
    float    min_peak_radius                 = my_current_job.arguments[4].ReturnFloatArgument( );
    float    pixel_size                      = my_current_job.arguments[5].ReturnFloatArgument( );
    bool     read_coordinates                = my_current_job.arguments[6].ReturnBoolArgument( );
    int      mip_x_dimension                 = my_current_job.arguments[7].ReturnIntegerArgument( );
    int      mip_y_dimension                 = my_current_job.arguments[8].ReturnIntegerArgument( );
    int      result_number                   = my_current_job.arguments[9].ReturnIntegerArgument( );
    int      ignore_N_pixels_from_the_border = my_current_job.arguments[10].ReturnIntegerArgument( );

    float padding = 2.0f;

    Image mip_image;

    Peak current_peak;

    AnglesAndShifts angles;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size;

    int   number_of_peaks_found = 0;
    int   slab_thickness_in_pixels;
    int   binned_dimension_3d;
    float binned_pixel_size;
    float max_density;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  text_file_access_type;
    int   i, j;

    float coordinates[3];
    if ( read_coordinates )
        text_file_access_type = OPEN_TO_READ;
    else
        text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 4);
    if ( ! read_coordinates ) {
        coordinate_file.WriteCommentLine("          X              Y            Peak");

        mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), result_number);

        mip_x_dimension = mip_image.logical_x_dimension;
        mip_y_dimension = mip_image.logical_y_dimension;

        min_peak_radius = powf(min_peak_radius, 2);
    }

    if ( ignore_N_pixels_from_the_border > 0 && (ignore_N_pixels_from_the_border > mip_image.logical_x_dimension / 2 || ignore_N_pixels_from_the_border > mip_image.logical_y_dimension / 2) ) {
        wxPrintf("You have entered %d for ignore_N_pixels_from_the_border, which is too large given image half dimesnsions of %d (X) and %d (Y)",
                 ignore_N_pixels_from_the_border, mip_x_dimension / 2, mip_y_dimension / 2);
        exit(-1);
    }
    wxPrintf("\n");
    while ( 1 == 1 ) {
        if ( ! read_coordinates ) {
            // look for a peak..

            current_peak = mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, ignore_N_pixels_from_the_border);
            if ( current_peak.value < wanted_threshold )
                break;

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            address = 0;

            current_peak.x = current_peak.x + mip_image.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + mip_image.physical_address_of_box_center_y;

            //			wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

            for ( j = 0; j < mip_y_dimension; j++ ) {
                sq_dist_y = float(pow(j - current_peak.y, 2));
                for ( i = 0; i < mip_x_dimension; i++ ) {
                    sq_dist_x = float(pow(i - current_peak.x, 2));

                    // The square centered at the pixel
                    if ( sq_dist_x + sq_dist_y <= min_peak_radius ) {
                        mip_image.real_values[address] = -FLT_MAX;
                    }

                    address++;
                }
                address += mip_image.padding_jump_value;
            }

            coordinates[0] = current_peak.x * pixel_size;
            coordinates[1] = current_peak.y * pixel_size;
            //			coordinates[5] = binned_pixel_size * (slab.physical_address_of_box_center_z - binned_reconstruction.physical_address_of_box_center_z) - current_defocus;
            //			coordinates[5] = binned_pixel_size * slab.physical_address_of_box_center_z - current_defocus;
            coordinates[2] = current_peak.value;
            coordinate_file.WriteLine(coordinates);
        }

        wxPrintf("Peak %4i at x, y = %12.6f, %12.6f : %10.6f\n", current_peak.x * pixel_size, current_peak.y * pixel_size, current_peak.value);

        if ( read_coordinates && coordinate_file.number_of_lines == number_of_peaks_found )
            break;
    }

    // save the output image

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Template Results: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
