#include "gpu_core_headers.h"

#define DO_HISTOGRAM true

//__global__ void UpdateMipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel, __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks);

__global__ void UpdateMipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel,
                                         __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks);

TemplateSnrRatioCore::TemplateSnrRatioCore( ){

};

TemplateSnrRatioCore::TemplateSnrRatioCore(int number_of_jobs) {

    Init(number_of_jobs);
};

TemplateSnrRatioCore::~TemplateSnrRatioCore( ){

        // FIXME
        //	if (is_allocated_cummulative_histogram)
        //	{
        //		cudaErr(cudaFree((void *)cummulative_histogram));
        //		cudaErr(cudaFreeHost((void *)h_cummulative_histogram));
        //	}

};

void TemplateSnrRatioCore::Init(int number_of_jobs) {
    this->nThreads                        = 1;
    this->number_of_jobs_per_image_in_gui = 1;
    this->nGPUs                           = 1;
};

void TemplateSnrRatioCore::Init(MyApp*           parent_pointer,
                                Image&           input_reconstruction_correct,
                                Image&           input_reconstruction_wrong,
                                Image&           montage_image,
                                Image&           current_projection_correct,
                                Image&           current_projection_wrong,
                                float            psi_max,
                                float            psi_start,
                                float            psi_step,
                                AnglesAndShifts& angles,
                                EulerSearch&     global_euler_search,
                                int              first_search_position,
                                int              last_search_position,
                                ProgressBar*     my_progress,
                                int              number_of_rotations,
                                long             total_correlation_positions,
                                long             total_correlation_positions_per_thread,
                                wxString         data_directory_name)

{
    this->first_search_position = first_search_position;
    this->last_search_position  = last_search_position;
    this->angles                = angles;
    this->global_euler_search   = global_euler_search;

    this->data_directory_name = data_directory_name;

    this->psi_start = psi_start;
    this->psi_step  = psi_step;
    this->psi_max   = psi_max;

    this->number_of_rotations                    = number_of_rotations;
    this->total_correlation_positions            = total_correlation_positions;
    this->total_correlation_positions_per_thread = total_correlation_positions_per_thread;

    // 3D volumes for particle, correct template and incorrect template
    this->input_reconstruction_correct.CopyFrom(&input_reconstruction_correct); // TODO make sure these are pad-enabled
    this->input_reconstruction_wrong.CopyFrom(&input_reconstruction_wrong);

    this->montage_image.CopyFrom(&montage_image);
    this->current_projection_wrong.CopyFrom(&current_projection_wrong);
    this->current_projection_correct.CopyFrom(&current_projection_correct);

    d_montage_image.Init(this->montage_image);
    d_montage_image.CopyHostToDevice( );

    // FIXME & TODO: figure out what should be copied to device and what should be created on device
    d_current_projection_wrong.Init(this->current_projection_wrong);
    d_current_projection_correct.Init(this->current_projection_correct);

    d_padded_reference_correct.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true); // TODO may need to include padding
    d_padded_reference_wrong.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true); // TODO may need to include padding

    d_max_intensity_projection_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_max_intensity_projection_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);

    // d_best_psi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    // d_best_theta.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    // d_best_phi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);

    // single-view images
    d_sum1_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sum1_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sum2_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sum2_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sum3_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sum3_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);

    d_sumSq1_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sumSq1_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sumSq2_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sumSq2_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sumSq3_ac.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);
    d_sumSq3_cc.Allocate(d_montage_image.dims.x, d_montage_image.dims.y, d_montage_image.dims.z, true);

    this->my_progress    = my_progress;
    this->parent_pointer = parent_pointer;

    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev

    // Transfer the input image_memory_should_not_be_deallocated

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
};

void TemplateSnrRatioCore::RunInnerLoop(Image& projection_filter, int threadIDX, long& current_correlation_position) {

    // Make sure we are starting with zeros
    // d_best_psi.Zeros( );
    // d_best_phi.Zeros( );
    // d_best_theta.Zeros( );

    d_sum1_ac.Zeros( );
    d_sumSq1_ac.Zeros( );
    d_sum2_ac.Zeros( );
    d_sumSq2_ac.Zeros( );
    d_sum3_ac.Zeros( );
    d_sumSq3_ac.Zeros( );

    d_sum1_cc.Zeros( );
    d_sumSq1_cc.Zeros( );
    d_sum2_cc.Zeros( );
    d_sumSq2_cc.Zeros( );
    d_sum3_cc.Zeros( );
    d_sumSq3_cc.Zeros( );

    d_max_intensity_projection_ac.Zeros( );
    d_max_intensity_projection_cc.Zeros( );

    d_padded_reference_correct.Zeros( );
    d_padded_reference_wrong.Zeros( );

    d_montage_image.ConvertToHalfPrecision(false);
    d_padded_reference_correct.ConvertToHalfPrecision(false);
    d_padded_reference_wrong.ConvertToHalfPrecision(false);

    total_number_of_cccs_calculated = 0;

    // Either do not delete the single precision, or add in a copy here so that each loop over defocus vals
    // have a copy to work with. Otherwise this will not exist on the second loop

    cudaErr(cudaMalloc((void**)&my_peaks_ac, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_peaks_cc, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_new_peaks_ac, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_new_peaks_cc, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_stats_ac, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_stats_cc, sizeof(__half2) * d_montage_image.real_memory_allocated));

    // why original code works even if we don't zero my_stats at initialization?
    cudaErr(cudaMemset(my_stats_ac, 0, sizeof(__half2) * d_montage_image.real_memory_allocated));
    cudaErr(cudaMemset(my_stats_cc, 0, sizeof(__half2) * d_montage_image.real_memory_allocated));

    cudaEvent_t ref_correct_projection_is_free_Event, ref_wrong_projection_is_free_Event,
            gpu_work_is_done_Event;
    cudaErr(cudaEventCreateWithFlags(&ref_correct_projection_is_free_Event, cudaEventDisableTiming));
    cudaErr(cudaEventCreateWithFlags(&ref_wrong_projection_is_free_Event, cudaEventDisableTiming));
    cudaErr(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));

    int   ccc_counter = 0;
    int   current_search_position;
    float average_on_edge;
    float average_of_reals;
    float temp_float;
    float current_psi;
    float variance;

    int thisDevice;
    cudaGetDevice(&thisDevice);
    wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {

        if ( current_search_position % 10 == 0 ) {
            wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);
        }

        for ( float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
            // first template
            input_reconstruction_correct.ExtractSlice(current_projection_correct, angles, 1.0f, false);
            current_projection_correct.complex_values[0] = 0.0f + I * 0.0f;

            current_projection_correct.SwapRealSpaceQuadrants( );
            current_projection_correct.MultiplyPixelWise(projection_filter);
            current_projection_correct.BackwardFFT( );
            average_on_edge  = current_projection_correct.ReturnAverageOfRealValuesOnEdges( );
            average_of_reals = current_projection_correct.ReturnAverageOfRealValues( ) - average_on_edge;

            // Make sure the device has moved on to the padded projection
            cudaStreamWaitEvent(cudaStreamPerThread, ref_correct_projection_is_free_Event, 0);

            //// TO THE GPU ////
            d_current_projection_correct.CopyHostToDevice( );

            d_current_projection_correct.AddConstant(-average_on_edge);

            // The average in the full padded image will be different;
            average_of_reals *= ((float)d_current_projection_correct.number_of_real_space_pixels / (float)d_padded_reference_correct.number_of_real_space_pixels);

            d_current_projection_correct.MultiplyByConstant(rsqrtf(d_current_projection_correct.ReturnSumOfSquares( ) / (float)d_padded_reference_correct.number_of_real_space_pixels - (average_of_reals * average_of_reals)));
            d_current_projection_correct.ClipInto(&d_padded_reference_correct, 0, false, 0, 0, 0, 0);
            cudaEventRecord(ref_correct_projection_is_free_Event, cudaStreamPerThread);

            // second template
            input_reconstruction_wrong.ExtractSlice(current_projection_wrong, angles, 1.0f, false);
            current_projection_wrong.complex_values[0] = 0.0f + I * 0.0f;

            current_projection_wrong.SwapRealSpaceQuadrants( );
            current_projection_wrong.MultiplyPixelWise(projection_filter);
            current_projection_wrong.BackwardFFT( );
            average_on_edge  = current_projection_wrong.ReturnAverageOfRealValuesOnEdges( );
            average_of_reals = current_projection_wrong.ReturnAverageOfRealValues( ) - average_on_edge;

            // Make sure the device has moved on to the padded projection
            cudaStreamWaitEvent(cudaStreamPerThread, ref_wrong_projection_is_free_Event, 0);

            //// TO THE GPU ////
            d_current_projection_wrong.CopyHostToDevice( );

            d_current_projection_wrong.AddConstant(-average_on_edge);

            // The average in the full padded image will be different;
            average_of_reals *= ((float)d_current_projection_wrong.number_of_real_space_pixels / (float)d_padded_reference_wrong.number_of_real_space_pixels);

            d_current_projection_wrong.MultiplyByConstant(rsqrtf(d_current_projection_wrong.ReturnSumOfSquares( ) / (float)d_padded_reference_wrong.number_of_real_space_pixels - (average_of_reals * average_of_reals)));
            d_current_projection_wrong.ClipInto(&d_padded_reference_wrong, 0, false, 0, 0, 0, 0);
            cudaEventRecord(ref_wrong_projection_is_free_Event, cudaStreamPerThread);

            // For the cpu code (MKL and FFTW) the image is multiplied by N on the forward xform, and subsequently normalized by 1/N
            // cuFFT multiplies by 1/root(N) forward and then 1/root(N) on the inverse. The input image is done on the cpu, and so has no scaling.
            // Stating false on the forward FFT leaves the ref = ref*root(N). Then we have root(N)*ref*input * root(N) (on the inverse) so we need a factor of 1/N to come out proper. This is included in BackwardFFTAfterComplexConjMul
            d_padded_reference_correct.ForwardFFT(false);
            d_padded_reference_wrong.ForwardFFT(false);

            //      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);
            d_padded_reference_correct.BackwardFFTAfterComplexConjMul(d_montage_image.complex_values_16f, true);
            d_padded_reference_wrong.BackwardFFTAfterComplexConjMul(d_montage_image.complex_values_16f, true);

            this->MipPixelWise(__float2half_rn(current_psi), __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]),
                               __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]));

            ccc_counter++;
            total_number_of_cccs_calculated++;

            if ( ccc_counter % 10 == 0 ) {
                this->UpdateSums(my_stats_ac, d_sum1_ac, d_sumSq1_ac);
                this->UpdateSums(my_stats_cc, d_sum1_cc, d_sumSq1_cc);
            }

            if ( ccc_counter % 100 == 0 ) {

                d_sum2_ac.AddImage(d_sum1_ac);
                d_sum1_ac.Zeros( );

                d_sumSq2_ac.AddImage(d_sumSq1_ac);
                d_sumSq1_ac.Zeros( );

                d_sum2_cc.AddImage(d_sum1_cc);
                d_sum1_cc.Zeros( );

                d_sumSq2_cc.AddImage(d_sumSq1_cc);
                d_sumSq1_cc.Zeros( );
            }

            if ( ccc_counter % 10000 == 0 ) {

                d_sum3_ac.AddImage(d_sum2_ac);
                d_sum2_ac.Zeros( );

                d_sumSq3_ac.AddImage(d_sumSq2_ac);
                d_sumSq2_ac.Zeros( );

                d_sum3_cc.AddImage(d_sum2_cc);
                d_sum2_cc.Zeros( );

                d_sumSq3_cc.AddImage(d_sumSq2_cc);
                d_sumSq2_cc.Zeros( );
            }

            current_projection_correct.is_in_real_space = false;
            current_projection_wrong.is_in_real_space   = false;

            d_padded_reference_correct.is_in_real_space = true;
            d_padded_reference_wrong.is_in_real_space   = true;

            cudaEventRecord(gpu_work_is_done_Event, cudaStreamPerThread);

            //			first_loop_complete = true;

            if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
                current_correlation_position++;
                if ( current_correlation_position > total_correlation_positions )
                    current_correlation_position = total_correlation_positions;
                my_progress->Update(current_correlation_position);
            }

        } // loop over psi angles

    } // end of outer loop euler sphere position
    /*
    for ( current_search_position_sampled_view = first_search_position_sampled_view; current_search_position_sampled_view <= last_search_position_sampled_view; current_search_position_sampled_view++ ) {
        if ( current_search_position_sampled_view % 10 == 0 ) {
            wxPrintf("Starting position %d/ %d\n", current_search_position_sampled_view, last_search_position_sampled_view);
        }
        for ( int j = 0; j < number_of_rotations_sampled_view; j++ ) { // check if the max is actually the psi_max CHECKME FIXME
            ccc_counter              = 0; // stores the tm ccs calculated
            current_psi_sampled_view = psi_start + j * psi_step_sampled_view;
            view_counter             = current_search_position_sampled_view * number_of_rotations_sampled_view + j;
            wxPrintf("worker %i starts on view %i\n\n", ReturnThreadNumberOfCurrentThread( ), view_counter);
            angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
            //			current_projection.SetToConstant(0.0f); // This also sets the FFT padding to zero

            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);

            current_projection_image.complex_values[0] = 0.0f + I * 0.0f;

            current_projection_image.SwapRealSpaceQuadrants( );
            current_projection_image.MultiplyPixelWise(projection_filter);
            current_projection_image.BackwardFFT( );

            current_projection_image.AddConstant(-avg_for_normalization);
            current_projection_image.DivideByConstant(std_for_normalization);
            wxPrintf("view %i after avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), current_projection_image.ReturnVarianceOfRealValues( ));
            current_projection_image.ForwardFFT( );
            current_projection_image.SwapRealSpaceQuadrants( );

            cudaStreamWaitEvent(cudaStreamPerThread, image_projection_is_free_Event, 0);
            d_current_projection_image.CopyHostToDevice( );

            d_current_projection_image.ConvertToHalfPrecision(false);

            // normalize using avg and std from all sampled views
            // d_current_projection_image.AddConstant(-avg_for_normalization);
            // d_current_projection_image.MultiplyByConstant(1 / std_for_normalization);
            // d_current_projection_image.ClipInto(&d_padded_image, 0, false, 0, 0, 0, 0);

            //  d_padded_image.ForwardFFT(false); // IMPORTANT CHEKCME scaling must set to false ow the output is only hald of the input image
            // FIXME TODO move fft into cpu?
            // d_padded_image.SwapRealSpaceQuadrants( );

            cudaErr(cudaMemset(my_peaks_ac, 0, sizeof(__half2) * d_current_projection_image.real_memory_allocated));
            cudaErr(cudaMemset(my_peaks_cc, 0, sizeof(__half2) * d_current_projection_image.real_memory_allocated));

            for ( current_search_position_tm = first_search_position_tm; current_search_position_tm <= last_search_position_tm; current_search_position_tm++ ) {
                for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
                    angles_tm.Init(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_tm, 0.0, 0.0);
                    // generate projection from testing template for tm
                    input_reconstruction_wrong.ExtractSlice(current_projection_other, angles_tm, 1.0f, false);
                    current_projection_other.complex_values[0] = 0.0f + I * 0.0f;
                    current_projection_other.SwapRealSpaceQuadrants( );
                    current_projection_other.MultiplyPixelWise(projection_filter);
                    current_projection_other.BackwardFFT( );
                    average_on_edge  = current_projection_other.ReturnAverageOfRealValuesOnEdges( );
                    average_of_reals = current_projection_other.ReturnAverageOfRealValues( ) - average_on_edge;

                    //  variance = current_projection_other.ReturnSumOfSquares( ) - powf(current_projection_other.ReturnAverageOfRealValues( ), 2);
                    cudaStreamWaitEvent(cudaStreamPerThread, ref_wrong_projection_is_free_Event, 0);
                    //// TO THE GPU ////
                    d_current_projection_other.CopyHostToDevice( );
                    d_current_projection_other.AddConstant(-average_on_edge);
                    average_of_reals *= ((float)d_current_projection_other.number_of_real_space_pixels / (float)d_padded_reference_wrong.number_of_real_space_pixels);
                    d_current_projection_other.MultiplyByConstant(rsqrtf(d_current_projection_other.ReturnSumOfSquares( ) / (float)d_padded_reference_wrong.number_of_real_space_pixels - (average_of_reals * average_of_reals)));

                    //current_projection_2.AddGaussianNoise(10.0f);
                    // Zeroing the central pixel is probably not doing anything useful...
                    // d_current_projection_other.ZeroCentralPixel( );

                    d_current_projection_other.ClipInto(&d_padded_reference_wrong, 0, false, 0, 0, 0, 0);
                    cudaEventRecord(ref_wrong_projection_is_free_Event, cudaStreamPerThread);

                    d_padded_reference_wrong.ForwardFFT(false);

                    input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles_tm, 1.0f, false);
                    current_projection_correct_template.complex_values[0] = 0.0f + I * 0.0f;
                    current_projection_correct_template.SwapRealSpaceQuadrants( );
                    current_projection_correct_template.MultiplyPixelWise(projection_filter);
                    current_projection_correct_template.BackwardFFT( );
                    average_on_edge = current_projection_correct_template.ReturnAverageOfRealValuesOnEdges( );

                    average_of_reals = current_projection_correct_template.ReturnAverageOfRealValues( ) - average_on_edge;
                    // variance         = current_projection_correct_template.ReturnSumOfSquares( ) - powf(current_projection_correct_template.ReturnAverageOfRealValues( ), 2);

                    cudaStreamWaitEvent(cudaStreamPerThread, ref_correct_projection_is_free_Event, 0);

                    //// TO THE GPU ////
                    d_current_projection_correct_template.CopyHostToDevice( );
                    d_current_projection_correct_template.AddConstant(-average_on_edge);
                    average_of_reals *= ((float)d_current_projection_correct_template.number_of_real_space_pixels / (float)d_padded_reference_correct.number_of_real_space_pixels);
                    d_current_projection_correct_template.MultiplyByConstant(rsqrtf(d_current_projection_correct_template.ReturnSumOfSquares( ) / (float)d_padded_reference_correct.number_of_real_space_pixels - (average_of_reals * average_of_reals)));

                    // d_current_projection_correct_template.MultiplyByConstant(rsqrtf(variance));
                    //current_projection_2.AddGaussianNoise(10.0f);

                    d_current_projection_correct_template.ClipInto(&d_padded_reference_correct, 0, false, 0, 0, 0, 0);
                    // Zeroing the central pixel is probably not doing anything useful...
                    // d_current_projection_other.ZeroCentralPixel( );
                    // FIXME is this necessary for each projection?
                    cudaEventRecord(ref_correct_projection_is_free_Event, cudaStreamPerThread);

                    d_padded_reference_correct.ForwardFFT(false);

                    d_padded_reference_correct.BackwardFFTAfterComplexConjMul(d_current_projection_image.complex_values_16f, true);
                    d_padded_reference_wrong.BackwardFFTAfterComplexConjMul(d_current_projection_image.complex_values_16f, true);

                    this->MipPixelWise(__float2half_rn(current_psi_tm), __float2half_rn(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1]), __float2half_rn(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0]));

                    cudaEventRecord(ref_wrong_projection_is_free_Event, cudaStreamPerThread);
                    ccc_counter++;
                    total_number_of_cccs_calculated++;

                    if ( ccc_counter % 10 == 0 ) {
                        this->UpdateSums(my_stats_ac, d_sum1_ac, d_sumSq1_ac);
                        this->UpdateSums(my_stats_cc, d_sum1_cc, d_sumSq1_cc);
                    }

                    if ( ccc_counter % 100 == 0 ) {
                        d_sum2_ac.AddImage(d_sum1_ac);
                        d_sum1_ac.Zeros( );

                        d_sum2_cc.AddImage(d_sum1_cc);
                        d_sum1_cc.Zeros( );

                        d_sumSq2_ac.AddImage(d_sumSq1_ac);
                        d_sumSq1_ac.Zeros( );

                        d_sumSq2_cc.AddImage(d_sumSq1_cc);
                        d_sumSq1_cc.Zeros( );
                    }

                    if ( ccc_counter % 10000 == 0 ) { // TODO DEBUF AFTER
                        d_sum3_ac.AddImage(d_sum2_ac);
                        d_sum2_ac.Zeros( );

                        d_sum3_cc.AddImage(d_sum2_cc);
                        d_sum2_cc.Zeros( );

                        d_sumSq3_ac.AddImage(d_sumSq2_ac);
                        d_sumSq2_ac.Zeros( );

                        d_sumSq3_cc.AddImage(d_sumSq2_cc);
                        d_sumSq2_cc.Zeros( );
                    }

                    current_projection_other.is_in_real_space            = false;
                    current_projection_correct_template.is_in_real_space = false;
                    d_padded_reference_correct.is_in_real_space          = true;
                    d_padded_reference_wrong.is_in_real_space            = true;

                    cudaEventRecord(current_tm_is_done, cudaStreamPerThread);

                } // loop over tm psi angles
            } //loop over tm euler sphere position

            current_projection_image.is_in_real_space   = false;
            d_current_projection_image.is_in_real_space = false;

            cudaEventRecord(image_projection_is_free_Event, cudaStreamPerThread);
            cudaStreamWaitEvent(cudaStreamPerThread, current_tm_is_done, 0);

            wxPrintf("worker %i finished view %i total number %d\n\n", ReturnThreadNumberOfCurrentThread( ), view_counter, ccc_counter);

            // starting from here DEBUG PRIORITY

            this->UpdateSums(my_stats_ac, d_sum1_ac, d_sumSq1_ac);
            this->UpdateSums(my_stats_cc, d_sum1_cc, d_sumSq1_cc);
            // starting from here DEBUG PRIORITY

            d_sum2_ac.AddImage(d_sum1_ac); // AddImageBySlice not working; memory messed up; try to use single slice images instead;
            d_sumSq2_ac.AddImage(d_sumSq1_ac);

            d_sum3_ac.AddImage(d_sum2_ac);
            d_sumSq3_ac.AddImage(d_sumSq2_ac);

            d_sum2_cc.AddImage(d_sum1_cc);
            d_sumSq2_cc.AddImage(d_sumSq1_cc);

            d_sum3_cc.AddImage(d_sum2_cc);
            d_sumSq3_cc.AddImage(d_sumSq2_cc);
            // to here DEBUG

            
            d_sum2_ac.AddImage(d_sum1_ac);
            d_sumSq2_ac.AddImage(d_sumSq1_ac);

            d_sum3_ac.AddImage(d_sum2_ac);
            d_sumSq3_ac.AddImage(d_sumSq2_ac);

            d_sum2_cc.AddImage(d_sum1_cc);
            d_sumSq2_cc.AddImage(d_sumSq1_cc);

            d_sum3_cc.AddImage(d_sum2_cc);
            d_sumSq3_cc.AddImage(d_sumSq2_cc);

    this->WriteMipToImage( );
    d_max_intensity_projection_ac.QuickAndDirtyWriteSlices(wxString::Format("%s/d_mip_ac_view_%i_%f.mrc", data_directory_name, view_counter, psi_step_tm).ToStdString( ), 1, 1);
    d_max_intensity_projection_cc.QuickAndDirtyWriteSlices(wxString::Format("%s/d_mip_cc_view_%i_%f.mrc", data_directory_name, view_counter, psi_step_tm).ToStdString( ), 1, 1);

    if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
        current_correlation_position_sampled_view++;
        if ( current_correlation_position_sampled_view > total_correlation_positions_sampled_view_per_thread )
            current_correlation_position_sampled_view = total_correlation_positions_sampled_view_per_thread;
        my_progress->Update(current_correlation_position_sampled_view); // move progress bar to inside loop more informative this way?
    }
    cudaEventRecord(current_view_is_done, cudaStreamPerThread); //testing here
    d_max_intensity_projection_ac.CopyFrom2DImageTo3DImage(d_max_intensity_projection_ac_all_views, d_max_intensity_projection_ac_all_views.real_memory_allocated / d_max_intensity_projection_ac_all_views.dims.z * view_counter);
    d_max_intensity_projection_cc.CopyFrom2DImageTo3DImage(d_max_intensity_projection_cc_all_views, d_max_intensity_projection_cc_all_views.real_memory_allocated / d_max_intensity_projection_cc_all_views.dims.z * view_counter);

    d_sum3_ac.CopyFrom2DImageTo3DImage(d_sum_ac, d_sum_ac.real_memory_allocated / d_sum_ac.dims.z * view_counter);
    d_sum3_cc.CopyFrom2DImageTo3DImage(d_sum_cc, d_sum_cc.real_memory_allocated / d_sum_cc.dims.z * view_counter);
    d_sumSq3_ac.CopyFrom2DImageTo3DImage(d_sumSq_ac, d_sumSq_ac.real_memory_allocated / d_sumSq_ac.dims.z * view_counter);
    d_sumSq3_cc.CopyFrom2DImageTo3DImage(d_sumSq_cc, d_sumSq_cc.real_memory_allocated / d_sumSq_cc.dims.z * view_counter);

    //  d_max_intensity_projection_ac_all_views.QuickAndDirtyWriteSlices(wxString::Format("check_gpu_run/d_mip_all_views_%i.mrc", view_counter).ToStdString( ), 1, d_max_intensity_projection_ac_all_views.dims.z);
    //  d_sum_ac.QuickAndDirtyWriteSlices(wxString::Format("check_gpu_run/d_sum_ac_all_views_%i.mrc", view_counter).ToStdString( ), 1, d_sum_ac.dims.z);
    //  d_sumSq_ac.QuickAndDirtyWriteSlices(wxString::Format("check_gpu_run/d_sumSq_ac_all_views_%i.mrc", view_counter).ToStdString( ), 1, d_sum_ac.dims.z);

} // loop over sampled views psi angles

// there seems to be problem between the outer for loops connection
} // loop over sampled views euler sphere position
*/

    wxPrintf("\t\t\ntotal number %d\n", ccc_counter);

    cudaStreamWaitEvent(cudaStreamPerThread, gpu_work_is_done_Event, 0);

    this->UpdateSums(my_stats_ac, d_sum1_ac, d_sumSq1_ac);

    d_sum2_ac.AddImage(d_sum1_ac);
    d_sumSq2_ac.AddImage(d_sumSq1_ac);

    d_sum3_ac.AddImage(d_sum2_ac);
    d_sumSq3_ac.AddImage(d_sumSq2_ac);

    this->UpdateSums(my_stats_cc, d_sum1_cc, d_sumSq1_cc);

    d_sum2_cc.AddImage(d_sum1_cc);
    d_sumSq2_cc.AddImage(d_sumSq1_cc);

    d_sum3_cc.AddImage(d_sum2_cc);
    d_sumSq3_cc.AddImage(d_sumSq2_cc);

    this->MipToImage( );

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    cudaErr(cudaFree(my_peaks_ac));
    cudaErr(cudaFree(my_peaks_cc));

    cudaErr(cudaFree(my_stats_ac));
    cudaErr(cudaFree(my_stats_cc));

    cudaErr(cudaFree(my_new_peaks_ac));
    cudaErr(cudaFree(my_new_peaks_cc));

    // d_max_intensity_projection_ac.QuickAndDirtyWriteSlices(wxString::Format("debug/d_mip_ac_%i.mrc", ReturnThreadNumberOfCurrentThread( )).ToStdString( ), 1, 1);
    // d_max_intensity_projection_cc.QuickAndDirtyWriteSlices(wxString::Format("debug/d_mip_cc_%i.mrc", ReturnThreadNumberOfCurrentThread( )).ToStdString( ), 1, 1);
}

void TemplateSnrRatioCore::MipPixelWise(__half psi, __half theta, __half phi) {
    precheck

            // N*
            d_padded_reference_correct.ReturnLaunchParamtersLimitSMs(5.f, 1024);
    d_padded_reference_wrong.ReturnLaunchParamtersLimitSMs(5.f, 1024);

    UpdateMipPixelWiseKernel<<<d_padded_reference_correct.gridDims, d_padded_reference_correct.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference_correct.real_values_16f, my_peaks_ac, (int)d_padded_reference_correct.real_memory_allocated, psi, theta, phi, my_stats_ac, my_new_peaks_ac);
    UpdateMipPixelWiseKernel<<<d_padded_reference_wrong.gridDims, d_padded_reference_wrong.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference_wrong.real_values_16f, my_peaks_cc, (int)d_padded_reference_wrong.real_memory_allocated, psi, theta, phi, my_stats_cc, my_new_peaks_cc);

    postcheck
}

__global__ void UpdateMipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel,
                                         __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks) {

    //	Peaks tmp_peak;

    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x ) {

        const __half  half_val = correlation_output[i];
        const __half2 input    = __half2half2(half_val * __half(10000.0));
        const __half2 mulVal   = __halves2half2((__half)1.0, half_val);
        //    	my_stats[i].sum = __hadd(my_stats[i].sum, half_val);
        //    	my_stats[i].sq_sum = __hfma(__half(1000.)*half_val,half_val,my_stats[i].sq_sum);
        my_stats[i] = __hfma2(input, mulVal, my_stats[i]);
        //    	tmp_peak = my_peaks[i];
        //		const __half half_val = __float2half_rn(val);

        //			tmp_peak.psi = psi;
        //			tmp_peak.theta = theta;
        //			tmp_peak.phi = phi;
        if ( half_val > __low2half(my_peaks[i]) ) {
            //				tmp_peak.mip = half_val;
            my_peaks[i]     = __halves2half2(half_val, psi);
            my_new_peaks[i] = __halves2half2(theta, phi);

            //				my_peaks[i].mip = correlation_output[i];
            //				my_peaks[i].psi = psi;
            //				my_peaks[i].theta = theta;
            //				my_peaks[i].phi = phi;
        }
    }
    //
}

__global__ void WriteMipToImageKernel(const __half2*, const __half2* my_new_peaks, const int, cufftReal*);

void TemplateSnrRatioCore::MipToImage( ) {

    precheck
            dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3         gridDims        = dim3((d_max_intensity_projection_ac.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    WriteMipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_peaks_ac, my_new_peaks_ac, d_max_intensity_projection_ac.real_memory_allocated,
                                                                                 d_max_intensity_projection_ac.real_values_gpu);
    WriteMipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_peaks_cc, my_new_peaks_cc, d_max_intensity_projection_cc.real_memory_allocated,
                                                                                 d_max_intensity_projection_cc.real_values_gpu);
    postcheck
}

__global__ void WriteMipToImageKernel(const __half2* my_peaks, const __half2* my_new_peaks, const int numel, cufftReal* mip) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if ( x < numel ) {

        mip[x] = (cufftReal)__low2float(my_peaks[x]);
    }
}

__global__ void UpdateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum);

void TemplateSnrRatioCore::UpdateSums(__half2* my_stats, GpuImage& sum, GpuImage& sq_sum) {

    precheck
            dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3         gridDims        = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    UpdateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_stats, sum.real_memory_allocated, sum.real_values_gpu, sq_sum.real_values_gpu);
    postcheck
}

__global__ void UpdateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < numel ) {

        sum[x]    = __fmaf_rn(0.0001f, __low2float(my_stats[x]), sum[x]);
        sq_sum[x] = __fmaf_rn(0.0001f, __high2float(my_stats[x]), sq_sum[x]);

        my_stats[x] = __halves2half2((__half)0., (__half)0.);
    }
}

/*
void TemplateSnrRatioCore::MipPixelWise(__half psi, __half theta, __half phi) {

    precheck
            // N*
            d_padded_reference_correct.ReturnLaunchParamtersLimitSMs(5.f, 1024);
    d_padded_reference_wrong.ReturnLaunchParamtersLimitSMs(5.f, 1024);

    UpdateMipPixelWiseKernel<<<d_padded_reference_correct.gridDims, d_padded_reference_correct.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference_correct.real_values_16f, my_peaks_ac, (int)d_padded_reference_correct.real_memory_allocated, psi, theta, phi, my_stats_ac, my_new_peaks_ac);
    UpdateMipPixelWiseKernel<<<d_padded_reference_wrong.gridDims, d_padded_reference_wrong.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference_wrong.real_values_16f, my_peaks_cc, (int)d_padded_reference_wrong.real_memory_allocated, psi, theta, phi, my_stats_cc, my_new_peaks_cc);

    postcheck
}

__global__ void
UpdateMipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel, __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks) {

    //	Peaks tmp_peak;

    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x ) {

        const __half  half_val = correlation_output[i];
        const __half2 input    = __half2half2(half_val * __half(10000.0)); // cc*10000
        const __half2 mulVal   = __halves2half2((__half)1.0, half_val);

        my_stats[i] = __hfma2(input, mulVal, my_stats[i]); // TODO check top_K script

        if ( half_val > __low2half(my_peaks[i]) ) {
            //				tmp_peak.mip = half_val;
            my_peaks[i]     = __halves2half2(half_val, psi);
            my_new_peaks[i] = __halves2half2(theta, phi);
        }
    }
    //
}

__global__ void WriteMipToImageKernel(const __half2*, const __half2* my_new_peaks, const int, cufftReal*);

void TemplateSnrRatioCore::WriteMipToImage( ) {

    precheck
            dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3         gridDims        = dim3((d_max_intensity_projection_ac.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1); // CHECKME FIXME does gridDims matter? Do I really need to divide by z?

    WriteMipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_peaks_ac, my_new_peaks_ac, d_max_intensity_projection_ac.real_memory_allocated, d_max_intensity_projection_ac.real_values_gpu);
    WriteMipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_peaks_cc, my_new_peaks_cc, d_max_intensity_projection_cc.real_memory_allocated, d_max_intensity_projection_cc.real_values_gpu);

    postcheck
}

__global__ void WriteMipToImageKernel(const __half2* my_peaks, const __half2* my_new_peaks, const int numel, cufftReal* mip) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if ( x < numel ) {
        mip[x] = (cufftReal)__low2float(my_peaks[x]);
    }
}

__global__ void UpdateSumsKernel(__half2* temp_my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum);

void TemplateSnrRatioCore::UpdateSums(__half2* temp_my_stats, GpuImage& sum, GpuImage& sq_sum) {

    precheck
            dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3         gridDims        = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    UpdateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(temp_my_stats, sum.real_memory_allocated, sum.real_values_gpu, sq_sum.real_values_gpu);
    postcheck
}

__global__ void UpdateSumsKernel(__half2* temp_my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < numel ) {

        sum[x]    = __fmaf_rn(0.0001f, __low2float(temp_my_stats[x]), sum[x]);
        sq_sum[x] = __fmaf_rn(0.0001f, __high2float(temp_my_stats[x]), sq_sum[x]);

        temp_my_stats[x] = __halves2half2((__half)0., (__half)0.);
    }
}
*/